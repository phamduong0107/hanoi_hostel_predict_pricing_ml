"""Trọ Giá. Giao diện Streamlit dự đoán giá phòng trọ và trò chuyện với Gemini."""

import os
import sys
from calendar import monthrange
from datetime import date, datetime
from html import escape
from importlib import import_module
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for directory in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from feature_eng import normalize_listing_text, extract_amenities
from train_model import TEST_METRICS, LOCATION_COLUMNS, group_shap_contributions


AMENITIES = {
    "dieu_hoa": "Điều hòa", "nong_lanh": "Nóng lạnh", "ve_sinh_khep_kin": "Vệ sinh riêng",
    "day_du_noi_that": "Đủ nội thất", "thang_may": "Thang máy", "ban_cong": "Ban công",
    "cua_so": "Cửa sổ", "bep": "Bếp", "tu_lanh": "Tủ lạnh", "may_giat": "Máy giặt",
    "gac_xep": "Gác xép", "cho_de_xe": "Chỗ để xe", "wifi": "Wi-Fi",
    "gio_giac_tu_do": "Giờ giấc tự do",
}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
ROOM_EXAMPLES = (
    ("Cầu Giấy · 25 m²", "Quận Cầu Giấy", 25.0, ["dieu_hoa", "nong_lanh", "ve_sinh_khep_kin"]),
    ("Hoàng Mai · 18 m²", "Quận Hoàng Mai", 18.0, ["cho_de_xe"]),
    ("Hà Đông · 30 m²", "Quận Hà Đông", 30.0, ["day_du_noi_that", "ban_cong", "thang_may"]),
)


@st.cache_resource(show_spinner=False)
def load_resources():
    model = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
    locations = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_ready.csv"), usecols=LOCATION_COLUMNS)
    return model, locations.fillna("Chua_ro").drop_duplicates()


def reset_fields(*column):
    for name in column:
        st.session_state.pop(name, None)


def load_example(district, area, selected_amenities, today):
    reset_fields("ward", "street")
    st.session_state.update(
        district=district, area=area, amenities=selected_amenities, owner="Chưa rõ",
        posted_year=today.year, posted_month=today.month, posted_day=today.day,
        title=f"Phòng trọ {district.removeprefix('Quận ')}", description="",
    )


def format_vnd(price):
    return f"{price:,.0f}".replace(",", ".") + " ₫"


def build_room_features(model, room):
    features = extract_amenities(room["title"], room["description"])
    features.update(dict.fromkeys(room["amenities"], 1))
    if room["owner"] != "Chưa rõ":
        features["chung_chu"] = int(room["owner"] == "Chung chủ")
        features["rieng_chu"] = int(room["owner"] == "Không chung chủ")
    features.update(
        area=room["area"], area_was_missing=0,
        posted_year=room["date"].year, posted_month=room["date"].month,
        text=normalize_listing_text(f"{room['title']} {room['description']}"),
    )
    features.update(dict(zip(LOCATION_COLUMNS, [room["district"], room["ward"] or "Chua_ro", room["street"] or "Chua_ro"])))
    return pd.DataFrame([features], columns=model.feature_names_in_)


def predict_room_price(model, room):
    x = build_room_features(model, room)
    price = float(model.predict(x)[0])
    contributions, base_price = group_shap_contributions(model, x)
    return price, float(base_price[0]), contributions.iloc[0].to_dict()


def ask_gemini(question, history, result=""):
    try:
        api_key = getattr(import_module("app.api_key"), "GEMINI_API_KEY", "")
    except (ImportError, SyntaxError):
        api_key = ""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Chưa đọc được GEMINI_API_KEY trong app/api_key.py.")

    messages = [{"role": message["role"], "parts": [{"text": message["text"]}]} for message in history[-8:]]
    if result:
        question += "\n\nKết quả ước tính do người dùng đính kèm:\n" + result
    messages.append({"role": "user", "parts": [{"text": question}]})
    try:
        response = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key.strip()},
            json={
                "systemInstruction": {"parts": [{"text": (
                    "Bạn là trợ lý Trọ Giá. Trả lời bằng tiếng Việt, ngắn gọn và thực tế. "
                    "Tư vấn cách xem phòng, hỏi chủ trọ và hiểu giá ước tính từ CatBoost. "
                    "Khi giải thích một dự đoán, dùng số liệu người dùng đính kèm. "
                    "Nếu thiếu số liệu, hướng dẫn người dùng ước tính rồi chọn gửi kèm. "
                    "Không tự bịa giá hoặc tin đăng hiện tại."
                )}]},
                "contents": messages,
                "generationConfig": {"maxOutputTokens": 1024, "thinkingConfig": {"thinkingLevel": "minimal"}},
            },
            timeout=(5, 25),
            allow_redirects=False,
        )
    except requests.Timeout:
        raise ValueError("Gemini phản hồi quá lâu. Hãy thử gửi lại câu hỏi.") from None
    except requests.RequestException:
        raise ValueError("Chưa kết nối được Gemini. Bạn có thể thử lại sau.") from None

    notice = {
        400: "Gemini từ chối yêu cầu. Hãy kiểm tra quyền của khóa API.",
        401: "Gemini từ chối xác thực. Hãy kiểm tra khóa API.",
        403: "Khóa hiện tại chưa có quyền truy cập Gemini API.",
        429: "Gemini đang giới hạn lượt gọi hoặc đã hết hạn mức.",
        503: "Gemini đang quá tải. Hãy thử gửi lại sau ít phút.",
    }
    if not 200 <= response.status_code < 300:
        raise ValueError(notice.get(response.status_code, "Gemini chưa phản hồi. Hãy thử lại sau."))
    try:
        candidates = response.json().get("candidates") or []
        response_parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        answer = "\n".join(item["text"] for item in response_parts if item.get("text") and not item.get("thought")).strip()
    except (ValueError, AttributeError, TypeError):
        answer = ""
    if not answer:
        raise ValueError("Gemini chưa trả về câu trả lời. Hãy thử hỏi theo cách khác.")
    return answer


def render_app():
    st.set_page_config(page_title="Trọ Giá · Phòng trọ Hà Nội", page_icon="🏠", layout="wide")
    st.html("""
    <style>
    .stMainBlockContainer {max-width: 1180px; padding-top: 2.2rem}
    .tg-hero {position: relative; overflow: hidden; isolation: isolate; color: #fbf8ed;
        background: linear-gradient(115deg,#163c34,#165e4f); border-radius: 22px;
        padding: clamp(24px,4vw,48px); margin-bottom: 10px}
    .tg-hero h1 {color: #fbf8ed; font-size: clamp(2rem,4.8vw,3.7rem); line-height: 1.05;
        letter-spacing: -.055em; margin: .2rem 0 .65rem; padding: 0}
    .tg-hero p {max-width: 36rem; color: #dae9de; margin: 0; line-height: 1.55}
    .tg-hero svg {position:absolute; z-index:-1; right:2%; bottom:-2%; width:38%; max-height:100%; opacity:.35}
    .tg-tag {font-size: .73rem; letter-spacing: .15em; font-weight: 760; text-transform: uppercase}
    .tg-hero .tg-tag {color: #e5c489}
    .tg-panel {color:#214a3e; border-radius: 15px; border:1px solid #c9ded2; background:#e9f2eb; padding:20px}
    .tg-panel p {margin: 5px 0 0; color:#435e53}
    .tg-price {display:block; font-size:clamp(1.9rem,4vw,2.7rem); letter-spacing:-.055em;
        font-weight:780; line-height:1.12; margin:6px 0}
    .tg-lines {font-variant-numeric: tabular-nums}
    .tg-line {display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid #dce7de;
        padding:7px 0; font-size:.88rem}
    .tg-plus {color:#15624e}.tg-minus {color:#9a4030}
    .st-key-tt_form,.st-key-tt_result,.st-key-tt_chat {background:#fffefa; border-radius:16px}
    .st-key-tt_form h3,.st-key-tt_result h3,.st-key-tt_chat h3 {padding-top:.15rem; padding-bottom:.25rem}
    @media(max-width:620px) {.stMainBlockContainer{padding-top:1.2rem}.tg-hero svg{display:none}}
    </style>
    <section class="tg-hero">
      <div class="tg-tag">Trọ Giá · Hà Nội</div>
      <h1>Phòng này thuê bao nhiêu?</h1>
      <p>Chọn thông tin căn phòng để xem giá thuê tham khảo và hiểu các yếu tố tác động đến dự đoán.</p>
      <svg viewBox="0 0 420 260" aria-hidden="true" fill="none" stroke="#e5c489" stroke-width="2">
        <path d="M0 246H420M72 246V107L195 28L320 107V246M48 121L195 27L343 121M120 246V126H273V246M120 176H273M196 126V246M351 246v-83h44v83M20 246v-63h30v63"/>
        <path d="M140 144h37v17h-37zM215 144h37v17h-37zM140 193h37v32h-37zM215 193h37v32h-37zM181 82h30v27h-30z"/>
      </svg>
    </section>
    """)

    model, locations = load_resources()
    today = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
    district_options = sorted(set(locations["Quận/Huyện"]) - {"Chua_ro"})
    prediction_tab, chat_tab = st.tabs(["Ước tính giá thuê", "Chatbot Gemini"])
    context = ""

    with prediction_tab:
        st.caption("Thử một căn phòng mẫu hoặc điền các trường có dấu *.")
        for column, (name, example_district, example_area, example_amenities) in zip(st.columns(3), ROOM_EXAMPLES):
            column.button(name, width="stretch", on_click=load_example, args=(example_district, example_area, example_amenities, today))

        form_column, result_column = st.columns([1.45, 1], gap="large")
        with form_column, st.container(border=True, key="tt_form"):
            st.markdown("### 01 · Không gian và vị trí")
            room_type, area_column = st.columns(2)
            room_type.selectbox("Loại phòng *", ["Phòng trọ"], key="room_type")
            area = area_column.number_input(
                "Diện tích (m²) *", min_value=1.0, max_value=200.0, value=None, step=1.0,
                placeholder="Ví dụ 25", key="area",
            )
            district = st.selectbox(
                "Quận / Huyện *", district_options, index=None, placeholder="Chọn quận/huyện", key="district",
                on_change=reset_fields, args=("ward", "street"),
            )
            selected_area = locations.loc[locations["Quận/Huyện"].eq(district)]
            ward_options = sorted(set(selected_area["Phường/Xã"]) - {"Chua_ro"})
            ward_column, street_column = st.columns(2)
            ward = ward_column.selectbox(
                "Phường / Xã", ward_options, index=None, placeholder="Chọn phường/xã",
                key="ward", disabled=not ward_options, on_change=reset_fields, args=("street",),
            )
            if ward:
                selected_area = selected_area.loc[selected_area["Phường/Xã"].eq(ward)]
            street_options = sorted(set(selected_area["Đường"]) - {"Chua_ro"})
            street = street_column.selectbox(
                "Tên đường", street_options, index=None, placeholder="Chọn tên đường",
                key="street", disabled=not street_options,
            )
            st.caption("Phường/xã và tên đường có thể để trống. Danh sách theo địa danh trong dữ liệu huấn luyện.")

            st.divider()
            st.markdown("### 02 · Tiện ích và mô tả")
            selected_amenities = st.multiselect("Tiện ích có sẵn", list(AMENITIES), format_func=AMENITIES.get, key="amenities")
            owner_status = st.selectbox("Ở cùng chủ nhà", ["Chưa rõ", "Chung chủ", "Không chung chủ"], key="owner")
            with st.expander("Thêm tiêu đề hoặc mô tả phòng", expanded=False):
                title = st.text_input("Tiêu đề tin", max_chars=250, key="title")
                description = st.text_area("Mô tả phòng", max_chars=4000, key="description",
                                     placeholder="Ví dụ phòng mới xây, ngõ rộng, có cửa sổ.")

            st.divider()
            st.markdown("### 03 · Thời điểm đăng")
            year_column, month_column, day_column = st.columns(3)
            year = year_column.selectbox(
                "Năm đăng *", list(range(today.year, 2011, -1)), index=None,
                placeholder="Năm", key="posted_year", on_change=reset_fields, args=("posted_month", "posted_day"),
            )
            month_options = list(range(1, (today.month if year == today.year else 12) + 1)) if year else []
            month = month_column.selectbox(
                "Tháng đăng *", month_options, index=None, placeholder="Tháng", key="posted_month",
                disabled=not month_options, on_change=reset_fields, args=("posted_day",),
            )
            day_options = []
            if year and month:
                last_day = today.day if (year, month) == (today.year, today.month) else monthrange(year, month)[1]
                day_options = list(range(1, last_day + 1))
            day = day_column.selectbox(
                "Ngày đăng *", day_options, index=None, placeholder="Ngày", key="posted_day", disabled=not day_options,
            )

            missing_fields = []
            if area is None:
                missing_fields.append("diện tích")
            if not district:
                missing_fields.append("quận/huyện")
            if not (year and month and day):
                missing_fields.append("ngày đăng")
            if missing_fields:
                st.caption("Cần chọn " + ", ".join(missing_fields) + " để ước tính.")
            room = dict(
                area=area, district=district, ward=ward, street=street,
                amenities=selected_amenities, owner=owner_status, title=title, description=description,
                date=date(year, month, day) if year and month and day else None,
            )
            if st.button("Ước tính giá thuê", type="primary", width="stretch", disabled=bool(missing_fields)) and not missing_fields:
                with st.spinner("Đang tính giá thuê…"):
                    result = predict_room_price(model, room)
                    st.session_state["du_doan"] = (room, result)
                st.toast(f"Giá ước tính {format_vnd(result[0])}/tháng")

        with result_column, st.container(border=True, key="tt_result"):
            st.markdown("### Giá thuê tham khảo")
            prediction = st.session_state.get("du_doan")
            state_attributes = f'data-area="{area!r}" data-district="{escape(district or "", quote=True)}"' 
            if prediction and prediction[0] == room:
                price, base_price, contributions = prediction[1]
                st.html(f'<div class="tg-panel"><div class="tg-tag">Ước tính mỗi tháng</div>'
                        f'<strong class="tg-price" data-price="{price!r}" {state_attributes}>{format_vnd(price)}</strong><p>Phòng trọ tại Hà Nội</p></div>')
                st.caption("Khoản cộng hoặc trừ so với mức giá gốc của mô hình (SHAP).")
                row = [f'<div class="tg-line"><span>Mức giá gốc</span><strong>{format_vnd(base_price)}</strong></div>']
                for name, contribution in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True):
                    css_class = "tg-plus" if contribution >= 0 else "tg-minus"
                    sign = "+" if contribution >= 0 else "−"
                    row.append(f'<div class="tg-line"><span>{name}</span><strong class="{css_class}">{sign}{format_vnd(abs(contribution))}</strong></div>')
                st.html('<div class="tg-lines">' + "".join(row) + '</div>')
                context = (
                    f"{area:g} m² tại {district}. Giá ước tính {format_vnd(price)}/tháng. "
                    f"Mức giá gốc {format_vnd(base_price)}. Đóng góp SHAP "
                    + ", ".join(f"{name} {contribution:+.0f} đồng" for name, contribution in contributions.items()) + "."
                )
                st.caption("Bạn có thể gửi kèm kết quả này ở tab Chatbot Gemini.")
            else:
                notice = "Thông tin đã thay đổi. Bấm Ước tính giá thuê để cập nhật." if prediction else (
                    "Chọn một phòng mẫu hoặc điền thông tin rồi bấm Ước tính giá thuê."
                )
                st.html(f'<div class="tg-panel"><div class="tg-tag">Ước tính mỗi tháng</div>'
                        f'<strong class="tg-price" {state_attributes}>—</strong><p>{notice}</p></div>')

            with st.expander("Kết quả mô hình trên tập test"):
                metrics = TEST_METRICS
                st.caption(f"{metrics['so_tin']:,} tin từ {metrics['tu_ngay']}. Tập test đã dùng ở các vòng trước.".replace(",", "."))
                st.markdown(f"**R²** {metrics['R2']:.4f}".replace(".", ","))
                st.markdown(f"**RMSE** {format_vnd(metrics['RMSE'])}/tháng")
                st.markdown(f"**MAE** {format_vnd(metrics['MAE'])}/tháng")

    with chat_tab, st.container(border=True, key="tt_chat"):
        st.markdown("### Trợ lý thuê phòng")
        st.caption("Hỏi Gemini về tiền điện nước, điều cần kiểm tra khi xem phòng hoặc ý nghĩa của kết quả ước tính.")
        include_estimate = st.checkbox("Gửi kèm kết quả ước tính", disabled=not context)
        history = st.session_state.setdefault("lich_su_chat", [])
        chat_box = st.container(height=330, border=False, autoscroll=True)
        with chat_box:
            with st.chat_message("assistant", avatar="🏠"):
                st.markdown("Chào bạn! Bạn muốn hỏi điều gì về việc thuê phòng?")
            for message in history:
                with st.chat_message("assistant" if message["role"] == "model" else "user"):
                    st.markdown(message["text"])
        with st.form("gui_cau_hoi", clear_on_submit=True, border=False):
            question = st.text_area(
                "Câu hỏi của bạn", placeholder="Ví dụ: Khi xem phòng nên hỏi gì về tiền điện?",
                height=72, max_chars=1500,
            )
            submitted = st.form_submit_button("Gửi câu hỏi", type="primary")
        if submitted and not question.strip():
            st.caption("Hãy nhập câu hỏi để gửi cho Gemini.")
        if submitted and question.strip():
            question = question.strip()
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Gemini đang trả lời…"):
                            answer = ask_gemini(question, history, context if include_estimate else "")
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.markdown(answer)
                        history.extend([{"role": "user", "text": question}, {"role": "model", "text": answer}])
                        del history[:-16]


if __name__ == "__main__":
    render_app()
