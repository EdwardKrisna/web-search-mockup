# app.py
import streamlit as st
from openai import AsyncOpenAI
import googlemaps
import os
import pandas as pd
import asyncio
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine
import json
from datetime import datetime

# ----------------------------------------------------------
# 1.  SECRETS  →  CLIENTS  →  ENGINE
# ----------------------------------------------------------
@st.cache_resource
def init_clients():
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        google_key = st.secrets["GOOGLE_API_KEY"]
        db_user = st.secrets["DB_USER"]
        db_pwd  = st.secrets["DB_PASSWORD"]
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]

        gpt_async = AsyncOpenAI(api_key=openai_key)
        gmaps     = googlemaps.Client(key=google_key)
        engine    = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}"
        )
        return gpt_async, gmaps, engine
    except Exception as e:
        st.error(f"Kesalahan koneksi: {e}")
        st.stop()

gpt_client_async, gmaps_client, engine = init_clients()

# ----------------------------------------------------------
# 2.  AGENTIC VIEW  (100 % original logic)
# ----------------------------------------------------------
class AgenticView:
    def __init__(self, google_client, gpt_client_async, engine):
        self.google_client = google_client
        self.gpt_client_async = gpt_client_async
        self.engine = engine

    # -------------  all your methods unchanged  --------------
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
                st.error(f"Gagal mendapatkan koordinat: {e2}")
                return None
        parameter["longitude"] = lon
        parameter["latitude"]  = lat
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
        return gpd.GeoDataFrame([row], crs="EPSG:4326")

    async def find_neighbour(self, distance_m, lon, lat):
        sql = f"""
        WITH q AS (SELECT ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography AS pt)
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
            ST_Distance(t.geog, q.pt) AS distance_m
        FROM objek_penilaian t, q
        WHERE ST_DWithin(t.geog, q.pt, {distance_m})
          AND t.longitude <> 0
          AND ST_Distance(t.geog, q.pt) > 0
        ORDER BY distance_m
        LIMIT 20;
        """
        return pd.read_sql(sql, self.engine)

    async def _similarity_row(self, user_pemberi_tugas, user_tahun,
                              user_kepemilikan, user_dokumen_kepemilikan,
                              user_tujuan_penilaian, user_jenis_objek, row):
        CERT = """
        Dokumen kepemilikan levels (strongest → weakest):
        1  Sertifikat Hak Milik (SHM) = full ownership
        2  Sertifikat Hak Guna Bangunan (HGB) = right to build, upgradable to SHM
        3  Sertifikat Hak Pakai (SHP) = right to use, time-limited
        """
        instruksi = (
            f"{CERT}\n\n"
            "Compare the two short land-valuation records and answer with the single line: x% "
            "where x is an integer 0-100 expressing how likely these two records refer to the SAME object."
        )
        user = f"Pemberi tugas: {user_pemberi_tugas}, tahun: {user_tahun}, jenis objek: {user_jenis_objek}, kepemilikan: {user_kepemilikan}, dokumen: {user_dokumen_kepemilikan}, tujuan: {user_tujuan_penilaian}"
        db   = f"Pemberi tugas: {row['pemberi_tugas']}, tahun: {row['tahun_kontrak']}, jenis objek: {row['jenis_objek_text']}, kepemilikan: {row['kepemilikan']}, dokumen: {row['dokumen_kepemilikan']}, tujuan: {row['tujuan_penugasan_text']}"
        resp = await self.gpt_client_async.responses.create(
            model="gpt-4.1-mini",
            instructions=instruksi,
            input=f"User: {user}\nDatabase: {db}"
        )
        return resp.output_text.strip().splitlines()[0]

    async def _add_similarity_column(self, neighbour_df, parameter):
        if neighbour_df.empty:
            neighbour_df["similarity_pct"] = []
            return neighbour_df

        tasks = [
            self._similarity_row(
                parameter.get("pemberi_tugas", ""),
                parameter.get("tahun", 0),
                parameter.get("kepemilikan", ""),
                parameter.get("dokumen_kepemilikan", ""),
                parameter.get("tujuan_penilaian", ""),
                parameter.get("jenis_objek", ""),
                row
            ) for _, row in neighbour_df.iterrows()
        ]
        neighbour_df["similarity_pct"] = await asyncio.gather(*tasks)
        return neighbour_df

    async def get_llm_response_of_object(self, df, gdf_from_params):
        fetched = df.to_json(orient="records") if df is not None else None
        prospect = gdf_from_params.to_dict(orient="records")
        prompt = f"""
            You are a speaking assistant tasked with assisting an assessment firm to:
            1. Prevent conflicts of interest
            2. Avoid duplication of work
            3. Avoid re-evaluating the same object

            The following is data on new assignment prospects :
            {prospect}

            And the following is data on assignments previously performed by the firm :
            {fetched}

            The similarity_pct column shows the level of similarity (0-100%) between a user object and an object in the database.
            Your task:
            - Compare each object in prospected_jobs with the list in fetched_context.
            - Identify any objects that are potentially identical, similar, or potentially conflicting.
            - Explain the reasons for the similarity (e.g., similar addresses, close coordinates, same assignor name, high similarity_pct value, etc.).

            NOTE : Use Bahasa Indonesia!
            """
        
        resp = await self.gpt_client_async.responses.create(model="gpt-4.1-mini", input=prompt)
        return resp.output_text

    # ---- OPTIONAL news search (set DO_NEWS = False to disable) ----
    DO_NEWS = False   # <--- toggle off web search

    async def get_llm_response_of_task_giver(self, task_giver: str) -> str:
        if not self.DO_NEWS:
            return "Aman!"          # skip news search completely
        prompt = f"""
            News about the '{task_giver}' case in Indonesia, create a list! 
            The case : corruption, scandal, or anything bad about the company.
            The output MUST in a list, and show ONLY the news and links (no need like 'Berikut adalah' or other, only show the list)!
            The company name MUST the same, DONT show other news! If you found no news about this company, just give this output : 'Aman!'
            NOTE : The company name MUST EXACTLY THE SAME and use Bahasa Indonesia.
            """
        resp = await self.gpt_client_async.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            tools=[{"type": "web_search"}]
        )
        return resp.output_text

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
            float(parameter["latitude"])
        )
        neighbour = await self._add_similarity_column(neighbour, parameter)

        # keep only >= 30 % similarity
        def pct_to_float(s):
            try:
                return float(str(s).replace("%", "").strip())
            except:
                return 0
        neighbour["sim_num"] = neighbour["similarity_pct"].apply(pct_to_float)
        neighbour = neighbour[neighbour["sim_num"] >= 30].sort_values("sim_num", ascending=False)
        neighbour = neighbour.drop(columns=["sim_num"])

        summary, sentiment = await asyncio.gather(
            self.get_llm_response_of_object(neighbour, gdf_from_params),
            self.get_llm_response_of_task_giver(parameter["pemberi_tugas"])
        )
        return neighbour.drop(columns=["geometry"], errors="ignore"), \
               await self._build_json_output(summary, sentiment)

# ----------------------------------------------------------
# 3.  CACHE  AGENTIC-VIEW  INSTANCE
# ----------------------------------------------------------
@st.cache_resource
def get_agentic_view(_gpt, _gmaps, _engine):
    return AgenticView(_gmaps, _gpt, _engine)

agentic_view = get_agentic_view(gpt_client_async, gmaps_client, engine)

# ----------------------------------------------------------
# 4.  SESSION STATE
# ----------------------------------------------------------
if "result_ready"   not in st.session_state:
    st.session_state.result_ready   = False
if "neighbour_df"   not in st.session_state:
    st.session_state.neighbour_df   = None
if "json_result"    not in st.session_state:
    st.session_state.json_result    = None
if "history"        not in st.session_state:
    st.session_state.history        = []

# ----------------------------------------------------------
# 5.  CSS  +  HEADER
# ----------------------------------------------------------
st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:2rem}
.info-box{background-color:#f0f2f6;padding:1rem;border-radius:.5rem;margin:1rem 0}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏢 Agentic View 1.0 – BETA</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# 6.  TABS
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Input Data", "📊 Hasil Analisis", "📜 Riwayat"])

# ========================================================== TAB 1
with tab1:
    st.header("Input Data Penugasan Baru")
    c1, c2 = st.columns(2)
    with c1:
        jenis_objek      = st.selectbox("Jenis Objek *", ["Rumah Tinggal","Ruko","Kantor","Pabrik","Tanah Kosong","Lainnya"])
        pemberi_tugas    = st.text_input("Pemberi Tugas *", placeholder="PT Bank ABC")
        tahun            = st.number_input("Tahun Kontrak *", min_value=2000, max_value=2100, value=datetime.now().year)
        tujuan_penilaian = st.text_input("Tujuan Penilaian *", value="Kredit Pemilikan Rumah")
        jenis_transaksi  = st.text_input("Jenis Transaksi", value="Jual-Beli")
    with c2:
        alamat_lokasi = st.text_area("Alamat Lokasi *", placeholder="Jl. Gandaria VI No 12, Jakarta Selatan", height=100)
        lon_col, lat_col = st.columns(2)
        with lon_col: longitude = st.text_input("Longitude", placeholder="opsional")
        with lat_col:  latitude  = st.text_input("Latitude",  placeholder="opsional")
        luas_tanah    = st.number_input("Luas Tanah (m²)", min_value=0, value=180, step=10)
        luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0, value=140, step=10)

    st.divider()
    bc1, bc2, _ = st.columns([1,1,2])
    with bc1:
        submit = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)
    with bc2:
        if st.button("🗑️ Bersihkan Form", use_container_width=True): st.rerun()

    if submit:
        if not (pemberi_tugas and alamat_lokasi):
            st.error("⚠️ Mohon isi semua field yang wajib (*)")
        else:
            param = {
                "longitude"        : float(longitude) if longitude else None,
                "latitude"         : float(latitude)  if latitude  else None,
                "jenis_objek"      : jenis_objek,
                "pemberi_tugas"    : pemberi_tugas,
                "nomor_kontrak"    : None,
                "luas_tanah"       : luas_tanah,
                "luas_bangunan"    : luas_bangunan,
                "tahun"            : tahun,
                "tujuan_penilaian" : tujuan_penilaian,
                "jenis_transaksi"  : jenis_transaksi,
                "alamat_lokasi"    : alamat_lokasi,
                "kepemilikan"      : "tunggal",
                "dokumen_kepemilikan":"Sertifikat Hak Milik"
            }
            with st.spinner("🔄 Sedang menganalisis…"):
                try:
                    n_df, js = asyncio.run(agentic_view.get_result(param))
                    st.session_state.neighbour_df = n_df
                    st.session_state.json_result  = js
                    st.session_state.result_ready = True
                    # ---- HISTORY ----
                    st.session_state.history.append({
                        "timestamp": datetime.now(),
                        "pemberi_tugas": pemberi_tugas,
                        "alamat": alamat_lokasi,
                        "results": js
                    })
                    st.success("✅ Analisis selesai! Lihat tab Hasil Analisis.")
                except Exception as e:
                    st.error(f"❌ Kesalahan: {e}")

# ========================================================== TAB 2
with tab2:
    st.header("Hasil Analisis")
    if st.session_state.result_ready:
        df = st.session_state.neighbour_df
        js  = st.session_state.json_result

        with st.expander("🗂️ Lihat Tabel Objek Serupa", expanded=True):
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("Tidak ditemukan objek serupa dalam radius 10 km.")

        st.markdown("---")
        st.subheader("📋 Ringkasan Analisis")
        st.markdown(f'<div class="info-box">{js["summary"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📰 Cek Pemberi Tugas")
        st.markdown(f'<div class="info-box">{js["client_sentiment"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        dc1, dc2 = st.columns(2)
        with dc1:
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Tabel (CSV)", csv,
                               f"similar_objects_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               "text/csv", use_container_width=True)
        with dc2:
            jsn = json.dumps(js, ensure_ascii=False, indent=2)
            st.download_button("📄 Download Laporan (JSON)", jsn,
                               f"analysis_report_{datetime.now():%Y%m%d_%H%M%S}.json",
                               "application/json", use_container_width=True)
    else:
        st.info("📝 Belum ada hasil analisis.")

# ========================================================== TAB 3
with tab3:
    st.header("Riwayat Analisis")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(f"📌 {item['timestamp']:%Y-%m-%d %H:%M:%S} – {item['pemberi_tugas']}"):
                st.write(f"**Alamat:** {item['alamat']}")
                st.write("**Analisis Pemberi Tugas:**")
                st.info(item["results"]["client_sentiment"])
                st.write("**Analisis Objek:**")
                st.info(item["results"]["summary"])
        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("📝 Belum ada riwayat analisis.")

# ----------------------------------------------------------
# 7.  FOOTER
# ----------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.85rem;'>"
    "Agentic View (BETA) v1.0 | Pastikan API keys sudah dikonfigurasi dengan benar</div>",
    unsafe_allow_html=True
)