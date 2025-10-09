import streamlit as st
from openai import AsyncOpenAI, OpenAI
import os
import pandas as pd
import asyncio
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine
import googlemaps
import json

# Import your AgenticView class
# from your_module import AgenticView

# Page configuration
st.set_page_config(
    page_title="Land Valuation Conflict Detection",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state for async handling
if 'result_ready' not in st.session_state:
    st.session_state.result_ready = False
if 'neighbour_df' not in st.session_state:
    st.session_state.neighbour_df = None
if 'json_result' not in st.session_state:
    st.session_state.json_result = None

# Title
st.title("🏠 Land Valuation Conflict Detection System")
st.markdown("---")

# # Sidebar for API credentials info
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.info("""
#     This application helps assessment firms:
#     - Prevent conflicts of interest
#     - Avoid duplication of work
#     - Avoid re-evaluating the same object
    
#     **Setup Required:**
#     Add these to `.streamlit/secrets.toml`:
#     ```toml
#     OPENAI_API_KEY = "your-key"
#     GOOGLE_API_KEY = "your-key"
#     DB_USER = "your-user"
#     DB_PASSWORD = "your-password"
#     DB_HOST = "your-host"
#     DB_PORT = "5432"
#     DB_NAME = "your-db"
#     ```
#     """)

# Initialize clients and engine
@st.cache_resource
def init_clients():
    """Initialize API clients and database engine"""
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]
        
        gpt_client_async = AsyncOpenAI(api_key=OPENAI_API_KEY)
        gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        
        return gpt_client_async, gmaps, engine
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        st.stop()

# Load clients
try:
    gpt_client_async, gmaps, engine = init_clients()
    st.sidebar.success("✅ APIs Connected")
except:
    st.sidebar.error("❌ Check your secrets.toml")
    st.stop()

# Copy your AgenticView class here or import it
# For now, I'll create a minimal version
class AgenticView:
    def __init__(self, google_client, gpt_client_async, engine):
        self.google_client = google_client
        self.gpt_client_async = gpt_client_async
        self.engine = engine
    
    def validate_locations(self, parameter: dict) -> dict:
        try:
            lon = float(parameter["longitude"])
            lat = float(parameter["latitude"])
        except Exception as e1:
            try:
                alamat = parameter["alamat_lokasi"]
                geocode_result = self.google_client.geocode(alamat)
                lat = float(geocode_result[0]['geometry']['location']['lat'])
                lon = float(geocode_result[0]['geometry']['location']['lng'])
            except Exception as e2:
                st.error(f"Failed to get coordinates: {e2}")
                return None
        parameter["longitude"] = lon
        parameter["latitude"] = lat
        return parameter
    
    async def create_gdf(self, parameter: dict) -> gpd.GeoDataFrame:
        row = {
            "longitude": parameter.get("longitude"),
            "latitude": parameter.get("latitude"),
            "jenis_objek": parameter.get("jenis_objek"),
            "pemberi_tugas": parameter.get("pemberi_tugas"),
            "nomor_kontrak": parameter.get("nomor_kontrak"),
            "tahun": parameter.get("tahun"),
            "luas_tanah": parameter.get("luas_tanah"),
            "luas_bangunan": parameter.get("luas_bangunan"),
            "tujuan_penilaian": parameter.get("tujuan_penilaian"),
            "jenis_transaksi": parameter.get("jenis_transaksi"),
            "alamat_lokasi": parameter.get("alamat_lokasi"),
            "geometry": Point(parameter.get("longitude"), parameter.get("latitude")),
        }
        gdf = gpd.GeoDataFrame([row], crs="EPSG:4326")
        return gdf
    
    async def find_neighbour(self, distance_m, lon, lat):
        query = f"""WITH q AS (
                        SELECT ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography AS pt
                    )
                    SELECT
                        t.pemberi_tugas,
                        t.jenis_objek_text,
                        t.cabang_text,
                        t.divisi,
                        t.tahun_kontrak,
                        t.alamat_lokasi,
                        t.keterangan,
                        t.kepemilikan,             
                        t.dokumen_kepemilikan,
                        t.tujuan_penugasan_text,
                        t.geometry,
                        ST_Distance(t.geog, q.pt) AS distance_m
                    FROM objek_penilaian t, q
                    WHERE ST_DWithin(t.geog, q.pt, {distance_m})
                      AND t.longitude <> 0
                      AND ST_Distance(t.geog, q.pt) > 0
                    ORDER BY distance_m
                    LIMIT 20;"""
        df = pd.read_sql(query, self.engine)
        return df
    
    async def _similarity_row(self, user_pemberi_tugas, user_tahun, 
                              user_kepemilikan, user_dokumen_kepemilikan, 
                              user_tujuan_penilaian, user_jenis_objek, row):
        CERT_SUMMARY = """
        Dokumen kepemilikan levels (strongest → weakest):
        1  Sertifikat Hak Milik (SHM) = full ownership
        2  Sertifikat Hak Guna Bangunan (HGB) = right to build, upgradable to SHM
        3  Sertifikat Hak Pakai (SHP) = right to use, time-limited
        """
        
        _SIMILARITY_INSTRUCT = (
            f"{CERT_SUMMARY}\n\n"
            "Compare the two short land-valuation records and answer with the single line: x% "
            "where x is an integer 0-100 expressing how likely these two records refer to the SAME object."
        )
        
        user_line = (f"""
            Pemberi tugas: {user_pemberi_tugas}, tahun: {user_tahun}, 
            jenis objek: {user_jenis_objek}, kepemilikan: {user_kepemilikan}, dokumen: {user_dokumen_kepemilikan}, 
            tujuan: {user_tujuan_penilaian}
            """
        )
        db_line = (f"""
            Pemberi tugas: {row['pemberi_tugas']}, tahun: {row['tahun_kontrak']}, 
            jenis objek: {row['jenis_objek_text']}, kepemilikan: {row['kepemilikan']}, dokumen: {row['dokumen_kepemilikan']}, 
            tujuan: {row['tujuan_penugasan_text']}
            """
        )
        
        resp = await self.gpt_client_async.responses.create(
            model="gpt-4.1-mini",
            instructions=_SIMILARITY_INSTRUCT,
            input=f"User: {user_line}\nDatabase: {db_line}",
        )
        return resp.output_text.strip().splitlines()[0]
    
    async def _add_similarity_column(self, neighbour_df, parameter):
        if neighbour_df.empty:
            neighbour_df["similarity_pct"] = []
            return neighbour_df
        
        user_pemberi_tugas = parameter.get("pemberi_tugas", "")
        user_tahun = parameter.get("tahun", 0)
        user_kepemilikan = parameter.get("kepemilikan", "")
        user_dokumen_kepemilikan = parameter.get("dokumen_kepemilikan", "")
        user_tujuan_penilaian = parameter.get("tujuan_penilaian", "")
        user_jenis_objek = parameter.get("jenis_objek", "")
        
        tasks = [
            self._similarity_row(user_pemberi_tugas, user_tahun, 
                               user_kepemilikan, user_dokumen_kepemilikan,
                               user_tujuan_penilaian, user_jenis_objek, row)
            for _, row in neighbour_df.iterrows()
        ]
        pct_list = await asyncio.gather(*tasks)
        neighbour_df["similarity_pct"] = pct_list
        return neighbour_df
    
    async def get_llm_response_of_object(self, df, gdf_from_params):
        fetched_context = df.drop('geometry', axis=1).to_json(orient="records") if df is not None else None
        prospected_jobs = gdf_from_params.to_dict(orient="records")
        
        response = await self.gpt_client_async.responses.create(
            model="gpt-4.1-mini",
            input=f"""
            You are a speaking assistant tasked with assisting an assessment firm to:
            1. Prevent conflicts of interest
            2. Avoid duplication of work
            3. Avoid re-evaluating the same object

            The following is data on new assignment prospects :
            {prospected_jobs}

            And the following is data on assignments previously performed by the firm :
            {fetched_context}

            The similarity_pct column shows the level of similarity (0-100%) between a user object and an object in the database.
            Your task:
            - Compare each object in prospected_jobs with the list in fetched_context.
            - Identify any objects that are potentially identical, similar, or potentially conflicting.
            - Explain the reasons for the similarity (e.g., similar addresses, close coordinates, same assignor name, high similarity_pct value, etc.).

            NOTE : Use Bahasa Indonesia!
            """
        )
        return response.output_text
    
    async def get_llm_response_of_task_giver(self, task_giver):
        response = await self.gpt_client_async.responses.create(
            model="gpt-4.1",
            input=f"""
            News about the '{task_giver}' case in Indonesia, create a list! 
            The case : corruption, scandal, or anything bad about the company.
            The output MUST in a list, and show ONLY the news and links (no need like 'Berikut adalah' or other, only show the list)!
            The company name MUST the same, DONT show other news! If you found no news about this company, just give this output : 'Aman!'
            NOTE : The company name MUST EXACTLY THE SAME and use Bahasa Indonesia.
            """,
            tools=[{"type": "web_search"}]
        )
        return response.output_text
    
    async def _build_json_output(self, summary, client_sentiment):
        return {"summary": summary, "client_sentiment": client_sentiment}
    
    async def get_result(self, parameter):
        parameter = self.validate_locations(parameter)
        if parameter is None:
            return None, None
            
        gdf_from_params = await self.create_gdf(parameter)
        neighbour = await self.find_neighbour(
            10000,
            float(parameter["longitude"]),
            float(parameter["latitude"]),
        )
        
        neighbour = await self._add_similarity_column(neighbour, parameter)
        
        # Sort by similarity_pct descending (highest similarity first)
        # Extract numeric value from 'x%' format
        def extract_pct(pct_str):
            try:
                return float(str(pct_str).replace('%', '').strip())
            except:
                return 0
        
        neighbour['similarity_numeric'] = neighbour['similarity_pct'].apply(extract_pct)
        neighbour = neighbour.sort_values('similarity_numeric', ascending=False)

        # Filter: only keep rows with similarity >= 30%
        neighbour = neighbour[neighbour['similarity_numeric'] >= 30]

        neighbour = neighbour.drop('similarity_numeric', axis=1)
        
        summary, client_sentiment = await asyncio.gather(
            self.get_llm_response_of_object(neighbour, gdf_from_params),
            self.get_llm_response_of_task_giver(parameter["pemberi_tugas"]),
        )
        
        json_out = await self._build_json_output(summary, client_sentiment)
        return neighbour, json_out

# Initialize AgenticView
@st.cache_resource
def get_agentic_view(_gpt_client_async, _gmaps, _engine):
    return AgenticView(_gmaps, _gpt_client_async, _engine)

agentic_view = get_agentic_view(gpt_client_async, gmaps, engine)

# Input form
st.header("📝 Input Data Objek Penilaian")

col1, col2 = st.columns(2)

with col1:
    longitude = st.text_input("Longitude", help="Leave empty to use geocoding from address")
    latitude = st.text_input("Latitude", help="Leave empty to use geocoding from address")
    jenis_objek = st.text_input("Jenis Objek", value="Rumah Tinggal")
    pemberi_tugas = st.text_input("Pemberi Tugas", value="PT. Bank Central Asia")
    nomor_kontrak = st.text_input("Nomor Kontrak (Optional)", value="")
    
with col2:
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0, value=180)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0, value=140)
    tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2025)
    tujuan_penilaian = st.text_input("Tujuan Penilaian", value="Kredit Pemilikan Rumah")
    jenis_transaksi = st.text_input("Jenis Transaksi", value="Jual-Beli")

alamat_lokasi = st.text_area("Alamat Lokasi", value="Jl. Gandaria VI No 12, Jakarta Selatan")

col3, col4 = st.columns(2)
with col3:
    kepemilikan = st.selectbox("Kepemilikan", ["tunggal", "bersama", "lainnya"])
with col4:
    dokumen_kepemilikan = st.selectbox(
        "Dokumen Kepemilikan",
        ["Sertifikat Hak Milik", "Sertifikat Hak Guna Bangunan", 
         "Sertifikat Hak Pakai", "Girik", "Petok D", "Letter C"]
    )

# Process button
if st.button("🔍 Analyze", type="primary"):
    with st.spinner("Processing... This may take a few minutes..."):
        # Prepare parameters
        params = {
            "longitude": float(longitude) if longitude else None,
            "latitude": float(latitude) if latitude else None,
            "jenis_objek": jenis_objek,
            "pemberi_tugas": pemberi_tugas,
            "nomor_kontrak": nomor_kontrak if nomor_kontrak else None,
            "luas_tanah": luas_tanah,
            "luas_bangunan": luas_bangunan,
            "tahun": tahun,
            "tujuan_penilaian": tujuan_penilaian,
            "jenis_transaksi": jenis_transaksi,
            "alamat_lokasi": alamat_lokasi,
            "kepemilikan": kepemilikan,
            "dokumen_kepemilikan": dokumen_kepemilikan,
        }
        
        # Run async function
        try:
            neighbour_df, json_result = asyncio.run(agentic_view.get_result(params))
            
            if neighbour_df is not None:
                st.session_state.neighbour_df = neighbour_df
                st.session_state.json_result = json_result
                st.session_state.result_ready = True
                st.success("✅ Analysis complete!")
            else:
                st.error("❌ Failed to process the request. Please check your inputs.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.result_ready:
    st.markdown("---")
    st.header("📊 Results")
    
    # Table of neighbors
    st.subheader("🗂️ Similar Objects Found")
    st.dataframe(
        st.session_state.neighbour_df,
        use_container_width=True,
        height=400
    )
    
    # Summary section
    st.markdown("---")
    st.subheader("📋 Analysis Summary")
    st.markdown(st.session_state.json_result["summary"])
    
    # Client sentiment (news)
    st.markdown("---")
    st.subheader("📰 Client Background Check")
    st.markdown(st.session_state.json_result["client_sentiment"])
    
    # JSON viewer (expander)
    with st.expander("🔍 See JSON Format"):
        st.json(st.session_state.json_result)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv = st.session_state.neighbour_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Table (CSV)",
            data=csv,
            file_name="similar_objects.csv",
            mime="text/csv"
        )
    with col2:
        json_str = json.dumps(st.session_state.json_result, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Report (JSON)",
            data=json_str,
            file_name="analysis_report.json",
            mime="application/json"
        )