"""Tạo dữ liệu đầu vào cho mô hình từ data_clean.csv."""

import re

import pandas as pd

from data_clean import strip_accents


CLEAN_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\data\data_clean.csv"
READY_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\data\data_ready.csv"

AMENITIES = {
    "dieu_hoa": r"dieu hoa|may lanh",
    "nong_lanh": r"nong lanh|binh nuoc nong",
    "thang_may": r"thang may",
    "ban_cong": r"ban cong",
    "ve_sinh_khep_kin": r"phong(?: tro)? k(?:h)?ep kin|(?:wc|ve sinh|nha tam)(?:\s*(?:va|\+)\s*bep)?\s+(?:k(?:h)?ep kin|rieng)",
    "tu_lanh": r"tu lanh",
    "bep": r"bep|cho nau an|khu nau an",
    "may_giat": r"may giat",
    "gac_xep": r"gac xep|gac lung",
    "cho_de_xe": r"cho de xe|cho gui xe|san de xe|nha de xe",
    "wifi": r"wi[\s-]?fi|internet",
    "day_du_noi_that": r"(?:day du|full)\s+(?:noi that|tien nghi|do)|noi that\s+(?:day du|full)|full do",
    "gio_giac_tu_do": r"gio giac (?:tu do|thoai mai)|(?:khong|ko|k) (?:han che|gioi han|quan ly) gio giac|di ve tu do",
}

QUALITY_SIGNALS = {
    "moi_xay": r"moi xay|vua xay|moi hoan thien|moi cai tao|moi sua",
    "cua_so": r"cua so",
    "mat_duong": r"mat duong|mat pho|mat tien",
    "noi_that_moi": r"noi that moi|do moi|trang bi moi",
}

NEARBY_PLACES = {
    "gan_truong": r"truong(?: hoc| dai hoc| cao dang| cap [123])?|dai hoc|cao dang",
    "gan_cho": r"cho(?!\s+(?:thue|de|gui|o|nau|lam)\b)|sieu thi",
    "gan_xe_buyt": r"(?:(?:tram|ben|diem) (?:xe )?|xe )?(?:buyt|bus)",
}

AMENITY_PATTERNS = {column: re.compile(r"\b(?:" + pattern + r")\b") for column, pattern in AMENITIES.items()}
AMENITY_ALTERNATIVES = r"\b(?:" + "|".join(AMENITIES.values()) + r")\b"
NEGATED_AMENITIES_PATTERN = re.compile(
    r"\b(?:khong|ko|k|chua)\s+(?:(?:co(?: san)?|lap|trang bi|duoc trang bi)\s+)?("
    + AMENITY_ALTERNATIVES
    + r"(?:\s*(?:[,/+&]|\bva\b|\bhoac\b)\s*" + AMENITY_ALTERNATIVES + r")*)"
)
SHARED_OWNER_PATTERN = r"(?:(?:o |song )?(?:chung|cung)(?: nha)?(?: voi)?|(?:o|song) voi) chu(?: nha)?"


NUMBER_WORD = r"(?:mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi|tram|linh|le|ruoi|phay)"
NUMBER_WORD_SEQUENCE = NUMBER_WORD + r"(?:\s+" + NUMBER_WORD + r"){0,7}"
TEXT_MASK_PATTERNS = tuple(map(re.compile, (
    r"\b" + NUMBER_WORD_SEQUENCE + r"\s+(?:trieu|tr|ty|nghin|ngan|dong)\b(?:\s+ruoi)?",
    r"\b(?:gia(?: thue| phong)?|tien phong)\s*[:=]?\s*" + NUMBER_WORD_SEQUENCE + r"\b",
    r"\S*\d\S*",
    r"\w*(?:trieu|nghin|vnd)\w*",
    r"[^\w\s]",
)))


def normalize_listing_text(text_input):
    """Che số tiền, điện thoại và các chuỗi chứa số trước khi học văn bản."""
    text = strip_accents(text_input)
    for pattern in TEXT_MASK_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip() or "khong_ro"


def extract_amenities(title, description):
    """Đổi thông tin được nhắc trong tin đăng thành các cờ 0/1."""
    text_input = strip_accents(title.replace("\n", "; ") + "; " + description.replace("\n", "; "))
    negated_text = "; ".join(hit.group(1) for hit in NEGATED_AMENITIES_PATTERN.finditer(text_input))
    features = {
        column: int(bool(pattern.search(text_input)) and not pattern.search(negated_text))
        for column, pattern in AMENITY_PATTERNS.items()
    }
    if re.search(r"\b(?:khong|ko|k|chua)\s+(?:ve sinh\s+)?k(?:h)?ep kin\b", text_input):
        features["ve_sinh_khep_kin"] = 0

    # “Chính chủ” chỉ nói người đăng là chủ nhà, không cho biết có ở cùng chủ.
    separate_owner = re.search(
        r"\b(?:(?:khong|ko|k)\s+" + SHARED_OWNER_PATTERN + r"|(?<!khong )(?<!ko )(?<!k )(?:rieng chu|tach biet (?:voi )?chu(?: nha)?))\b",
        text_input,
    )
    features["rieng_chu"] = int(bool(separate_owner))
    features["chung_chu"] = int(not separate_owner and bool(re.search(r"\b" + SHARED_OWNER_PATTERN + r"\b", text_input)))

    for column, place_pattern in NEARBY_PLACES.items():
        features[column] = int(bool(re.search(
            r"(?<!khong )(?<!ko )(?<!k )\b(?:gan|canh|sat|doi dien)\s+(?:(?:cac|nhieu|ngay|khu|mot)\s+)*(?:" + place_pattern + r")\b",
            text_input,
        )))
    for column, pattern in QUALITY_SIGNALS.items():
        mention = re.search(r"\b(?:" + pattern + r")\b", text_input)
        negation = re.search(r"\b(?:khong|ko|k|chua)\s+(?:(?:co|phai)\s+)?(?:" + pattern + r")\b", text_input)
        features[column] = int(bool(mention) and not negation)
    return features


def build_features(data):
    """Dựng bảng đặc trưng theo đúng thứ tự cột của mô hình."""
    location_columns = ["Quận/Huyện", "Phường/Xã", "Đường"]
    kept_columns = ["price", "area", *location_columns, "posted_date", "unit_group"]
    result = data[kept_columns].copy()

    # Giữ diện tích gốc và đánh dấu thiếu để bước train điền trung vị riêng trên train.
    result["area"] = pd.to_numeric(result["area"])
    result["area_was_missing"] = result["area"].isna().astype("int8")
    result["price"] = pd.to_numeric(result["price"])
    date_value = pd.to_datetime(result["posted_date"], utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
    if date_value.isna().any() or result["price"].isna().any():
        raise ValueError("Dữ liệu clean còn thiếu ngày đăng hoặc giá thuê.")

    result["posted_year"] = date_value.dt.year
    result["posted_month"] = date_value.dt.month
    for column in location_columns:
        name = result[column].fillna("").str.normalize("NFC").str.replace(r"\s+", " ", regex=True).str.strip()
        result[column] = name.mask(name.eq(""), "Chua_ro")

    text_pairs = list(
        data[["title", "description"]].fillna("").itertuples(index=False, name=None)
    )
    amenities = pd.DataFrame(
        [extract_amenities(title, description) for title, description in text_pairs],
        index=result.index,
        dtype="int8",
    )
    # posted_date và unit_group phục vụ chia tập, price là nhãn. Các cột còn lại là đầu vào.
    result = pd.concat([result, amenities], axis=1)
    result["text"] = [normalize_listing_text(title + " " + description) for title, description in text_pairs]
    return result


if __name__ == "__main__":
    result = build_features(pd.read_csv(CLEAN_FILE))
    result.to_csv(READY_FILE, index=False, encoding="utf-8-sig")
    print(f"Đã tạo {len(result):,} dòng, {result.shape[1]} cột tại {READY_FILE}.")
