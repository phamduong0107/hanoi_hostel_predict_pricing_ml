"""Làm sạch dữ liệu, chỉ giữ phòng trọ; ghi duy nhất data/data_clean.csv."""

import html
import re
import unicodedata
import pandas as pd
import numpy as np

RAW_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\data\data_raw.csv"
CLEAN_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\data\data_clean.csv"
MIN_RENT, MAX_RENT = 300_000, 30_000_000
MAX_AREA = 200


def strip_accents(value):
    text_value = str(value).casefold().replace("đ", "d")
    text_value = unicodedata.normalize("NFKD", text_value)
    text_value = re.sub(r"[\u0300-\u036f]", "", text_value)
    return re.sub(r"\s+", " ", text_value).strip().replace("–", "-").replace("—", "-")


def parse_number(value, kind):
    if pd.isna(value):
        return np.nan
    text_value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    if kind == "area":
        text_value = re.sub(r"m\s*\^?\s*2|mét\s+vuông|met\s+vuong", "", text_value).strip()
    else:
        text_value = re.sub(r"/\s*(?:tháng|thang)\b", "", text_value).strip()
        # Ví dụ: 3tr5 -> 3.5 triệu. Không tự đoán đơn vị khi chỉ có một số.
        text_value = re.sub(r"^(\d+)\s*(?:tr|triệu|trieu)\s*(\d+)$", r"\1.\2 triệu", text_value)
    pattern = r"(-?\d+(?:[.,]\d+)*)\s*(triệu|trieu|tr|tỷ|ty|nghìn|nghin|ngàn|ngan|k|đồng|dong|vnđ|vnd|đ|d|₫)?"
    result = re.fullmatch(pattern, text_value)
    if result is None:
        return np.nan  # Ví dụ: Thỏa thuận, 12-15 m², giá theo ngày hoặc nhiều mức giá.
    numeric_text = result.group(1)
    unit = result.group(2)
    if kind == "area" and unit is not None:
        return np.nan
    if re.fullmatch(r"-?\d{1,3}(?:[.,]\d{3})+", numeric_text):
        numeric_text = numeric_text.replace(".", "").replace(",", "")
    else:
        numeric_text = numeric_text.replace(",", ".")
    try:
        number = float(numeric_text)
    except ValueError:
        return np.nan
    if unit in ("triệu", "trieu", "tr"):
        number *= 1_000_000
    elif unit in ("tỷ", "ty"):
        number *= 1_000_000_000
    elif unit in ("nghìn", "nghin", "ngàn", "ngan", "k"):
        number *= 1_000
    return number if np.isfinite(number) else np.nan


def parse_date(value, crawled_at=""):
    text_value = str(value).strip()
    normalized_text = strip_accents(text_value)
    relative_date = re.fullmatch(r"(\d+)\s*(phut|gio|ngay|tuan|thang|nam)\s*truoc", normalized_text)
    if relative_date or normalized_text in ("hom nay", "hom qua"):
        reference_rows = pd.to_datetime(crawled_at, utc=True, errors="coerce")
        if pd.isna(reference_rows):
            return pd.NaT
        reference_rows = reference_rows.tz_convert("Asia/Ho_Chi_Minh")
        if not relative_date:
            return reference_rows - pd.Timedelta(days=int(normalized_text == "hom qua"))
        number, unit = int(relative_date.group(1)), relative_date.group(2)
        if unit == "thang":
            return reference_rows - pd.DateOffset(months=number)
        if unit == "nam":
            return reference_rows - pd.DateOffset(years=number)
        unit = {"phut": "m", "gio": "h", "ngay": "D", "tuan": "W"}[unit]
        return reference_rows - pd.to_timedelta(number, unit=unit)
    date_value = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text_value)
    if date_value:
        time_match = re.search(r"\b\d{1,2}:\d{2}\b", text_value)
        text_value = f"{date_value[3]}-{date_value[2]}-{date_value[1]} " + (time_match[0] if time_match else "00:00")
    date_value = pd.to_datetime(text_value, errors="coerce")
    if pd.isna(date_value):
        return pd.NaT
    if date_value.tzinfo is None:
        return date_value.tz_localize("Asia/Ho_Chi_Minh")
    return date_value.tz_convert("Asia/Ho_Chi_Minh")


def normalize_raw(raw):
    """Chuẩn hóa văn bản và gộp các bản ghi bổ sung của cùng tin."""
    data = raw.copy()
    data["recovery_note"] = raw.get("recovery_note", "")
    for column in ["title", "description", "address"]:
        data[column] = data[column].map(html.unescape).str.normalize("NFC")
        data[column] = data[column].str.replace(r"(?is)<(?:script|style)\b.*?</(?:script|style)>", " ", regex=True)
        data[column] = data[column].str.replace(r"(?i)<br\s*/?>|</p>", "\n", regex=True)
        data[column] = data[column].str.replace(r"<[^>]+>", " ", regex=True)
        data[column] = data[column].str.replace(r"\\[nr]", "\n", regex=True).str.replace("\xa0", " ")
        data[column] = data[column].str.replace(r"[ \t]+", " ", regex=True).str.strip()
    # Mã tin phải đúng quy tắc từng website; hậu tố "-2.html" không phải mã tin.
    for index, listing in data.iterrows():
        listing_id = ""
        if listing["source"] == "phongtro123.com":
            hit = re.search(r"-pr(\d+)\.html", listing["url"])
            hit = hit or re.search(r"\s+-\s+(\d{3,9})\s*$", listing["description"])
        elif listing["source"] == "phongtro.vn":
            hit = re.search(r"/(\d+)/?$", listing["url"])
        else:
            hit = re.search(r"-(\d+)\.html", listing["url"])
        if hit:
            listing_id = hit.group(1)
        if not listing["post_id"] and listing_id:
            data.at[index, "post_id"] = listing_id
            data.at[index, "recovery_note"] += "post_id:source;"
        if listing["source"] == "phongtro123.com" and (listing["post_id"] or listing_id):
            data.at[index, "description"] = re.sub(r"\s+-\s+" + re.escape(listing["post_id"] or listing_id) + r"\s*$", "", listing["description"])

    # Gộp bản bổ sung cùng tin/giá/ngày trước khi lọc: không để bản mô tả ngắn che lỗi bản đầy đủ.
    previous_count = len(data)
    for column in ["url", "post_id"]:
        has_key = data[column].ne("")
        is_duplicate = has_key & data.duplicated(["source", column, "price", "posted_date"], keep="last")
        data = data.loc[~is_duplicate].copy()
    print(f"Raw: {len(raw):,} dòng; gộp {previous_count - len(data):,} bản cùng tin/giá/ngày.")
    return data


def filter_rental_rooms(data):
    """Giữ tin cho thuê phòng trọ theo các dấu hiệu đã định."""
    title = data["title"].map(strip_accents)
    description = data["description"].map(strip_accents)
    count_before_filter = len(data)

    # Phòng trong nhà riêng vẫn có thể là phòng trọ; chỉ loại khi chào thuê cả nhà.
    room_pattern = r"\bphong(?!\s+(?:ngu|khach|tam|bep)\b)\b|\bnha tro\b"
    other_listing_pattern = r"\b(?:nguyen can|nguyen nha|ca nha|can ho|chung cu|ccmn|cc mini|studio|duplex|o ghep|ky tuc xa|ktx|sleep\s*box|mat bang|ki[ -]*ot|cua hang|van phong|sang nhuong|chuyen nhuong)\b"
    whole_house_pattern = r"\b(?:cho thue|can thue)\s+(?:ca\s+)?nha(?!\s+(?:tro|co phong|thua phong)\b)|\bbiet thu\b"
    has_room_signal = title.str.contains(room_pattern, regex=True)
    has_room_signal |= description.str.slice(0, 250).str.contains(r"\b(?:phong tro|nha tro|cho thue phong|phong khep kin)\b", regex=True)
    # "Gần chung cư" là tiện ích vị trí, không phải loại hình của phòng.
    title_for_filter = title.str.replace(r"\b(?:gan|canh|doi dien|sat|cach|khong phai)\s+(?:(?:khu|toa)\s+)?(?:chung cu|can ho|ccmn|studio|cua hang|van phong)\b", " dia diem ", regex=True)
    is_other_listing_type = title_for_filter.str.contains(other_listing_pattern, regex=True) | title_for_filter.str.contains(whole_house_pattern, regex=True)
    is_other_listing_type |= title.str.contains(r"\b(?:khach san|nha nghi|theo gio|theo ngay|theo dem)\b", regex=True)
    is_other_listing_type |= description.str.contains(r"\b(?:cho thue|sang nhuong|chuyen nhuong)\s+(?:1\s+|mot\s+)?(?:can ho|chung cu|studio|nha nguyen can|nguyen can|mat bang|cua hang|van phong)\b", regex=True)
    # Loại cả mô tả tự nhận là căn hộ/CCMN, nhưng bỏ qua ngữ cảnh "gần/không phải...".
    description_for_filter = description.str.replace(r"\b(?:gan|sat|canh|doi dien|cach|khong phai|khong o|khong cho thue)\b[^,;.!?]{0,35}\b(?:can ho|chung cu|ccmn|studio|nguyen can|o ghep|ky tuc xa|ktx)\b", " dia diem ", regex=True)
    is_other_listing_type |= description_for_filter.str.contains(r"\b(?:can ho|chung cu|ccmn|cc mini|studio|duplex|nguyen can|nguyen nha|o ghep|ky tuc xa|ktx|sleep\s*box)\b", regex=True)
    is_other_listing_type |= title.str.contains(r"\b(?:muon tim|can tim|tim thue|can thue)\b", regex=True)
    data = data.loc[has_room_signal & ~is_other_listing_type].copy()
    print(f"Giữ dấu hiệu phòng trọ: {len(data):,}; loại/không rõ loại hình: {count_before_filter - len(data):,}.")
    return data


def clean_price_and_area(data):
    """Khôi phục từ văn bản rồi áp dụng ngưỡng giá và diện tích."""
    for column in ("price", "area"):
        data[column] = data[column].apply(parse_number, kind=column)
    for column in ("price", "area"):
        data[column] = data[column].mask(data[column].le(0))
    price_pattern = r"(?<![\w.,])(\d+(?:[.,]\d+)*\s*(?:trieu|tr|ty|nghin|ngan|k|dong|vnd)\d*)(?!\w)"
    area_pattern = r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*m\s*2\b"
    price_range_pattern = r"\d+(?:[.,]\d+)?\s*(?:trieu|tr|k)?\s*(?:-|den|toi)\s*\d+(?:[.,]\d+)?\s*(?:trieu|tr|k)\b"
    area_range_pattern = r"\d+(?:[.,]\d+)?\s*(?:m2)?\s*(?:-|den|toi)\s*\d+(?:[.,]\d+)?\s*m2\b"
    for column in ["price", "area"]:
        recovered_count = 0
        for index in data.index[data[column].isna()]:
            candidate_numbers = []
            for text_index, text_input in enumerate([data.at[index, "title"], data.at[index, "description"]]):
                text_input = strip_accents(unicodedata.normalize("NFKC", text_input))
                range_pattern = price_range_pattern if column == "price" else area_range_pattern
                if re.search(range_pattern, text_input):
                    candidate_numbers = []
                    break  # Có khoảng giá/diện tích: giữ thiếu, không đoán một đầu khoảng.
                for result in re.finditer(price_pattern if column == "price" else area_pattern, text_input):
                    preceding_text = text_input[max(0, result.start() - 45):result.start()]
                    following_text = text_input[result.end():result.end() + 30]
                    if column == "price":
                        if re.search(r"\b(?:coc|dien|nuoc|phi|dich vu)\s*[:=]?\s*$", preceding_text):
                            continue
                        if re.match(r"\s*/\s*(?:nguoi|giuong|ngay|dem|gio|m2)\b", following_text):
                            continue
                        if text_index == 1 and not re.search(r"\b(?:gia(?: thue| phong)?|tien phong)\s*[:=]?\s*$", preceding_text) and not re.match(r"\s*/\s*thang\b", following_text):
                            continue
                    elif re.search(r"(?:[<>≤≥]\s*|\b(?:tren|hon|duoi|it nhat|toi da)\s*)$", preceding_text):
                        continue
                    elif re.search(r"\b(?:gac|gac xep|ban cong|san|mat tien|duong|ve sinh|wc|nha tam)\s*(?:rong|them)?\s*[:=]?\s*$", preceding_text):
                        continue
                    number = parse_number(result.group(1), column)
                    if pd.notna(number) and number > 0:
                        candidate_numbers.append(number)
            candidate_numbers = set(candidate_numbers)
            # Nhiều phòng: chỉ chọn khi mô tả ghép rõ từng diện tích với đúng giá đang ghi.
            if column == "area" and len(candidate_numbers) > 1 and pd.notna(data.at[index, "price"]):
                matched_pairs = re.findall(r"(?<![\d.,<>])(\d+(?:[.,]\d+)?)\s*m2\b(?:(?!\bva\b|m2|[;.!?]).){0,45}?\bgia\s*[:=]?\s*(\d+(?:[.,]\d+)?\s*(?:trieu|tr|nghin|ngan|k)\d*)\b", text_input)
                matching_areas = {parse_number(area_text, "area") for area_text, price in matched_pairs if parse_number(area_text, "area") in candidate_numbers and abs(parse_number(price, "price") - data.at[index, "price"]) < 1}
                if len(matching_areas) == 1:
                    candidate_numbers = matching_areas
                    data.at[index, "recovery_note"] += "area:explicit_room_price_pair;"
            if len(candidate_numbers) == 1:
                data.at[index, column] = candidate_numbers.pop()
                data.at[index, "recovery_note"] += f"{column}:source_text;"
                recovered_count += 1
        print(f"Khôi phục {column} từ text: {recovered_count:,} tin.")
    # Chỉ đánh dấu chênh lệch rõ: tiêu đề ghi "giá X", không ghi khoảng/từ, lệch >2 lần.
    conflicting_price = pd.Series(False, index=data.index)
    for index, listing in data.iterrows():
        text = strip_accents(listing["title"])
        if re.search(price_range_pattern, text):
            continue
        title_prices = re.findall(r"\bgia(?: thue| phong)?\s*[:=]?\s*(\d+(?:[.,]\d+)*\s*(?:trieu|tr|nghin|ngan|k)\d*)\b", text)
        if len(title_prices) == 1:
            price = parse_number(title_prices[0], "price")
            if pd.notna(price) and price > 0 and pd.notna(listing["price"]):
                conflicting_price.at[index] = max(price, listing["price"]) > 2 * min(price, listing["price"])
    print(f"Loại giá mâu thuẫn rõ giữa cột giá và tiêu đề (>2 lần): {conflicting_price.sum():,} tin.")
    data = data.loc[~conflicting_price].copy()
    previous_count = len(data)
    data = data.loc[data["price"].between(MIN_RENT, MAX_RENT)].copy()
    print(f"Loại giá thiếu/ngoài khoảng: {previous_count - len(data):,} tin.")
    previous_count = len(data)
    data = data.loc[~data["area"].gt(MAX_AREA)].copy()
    data["area_was_missing"] = data["area"].isna().astype(int)
    print(f"Loại diện tích >{MAX_AREA} m²: {previous_count - len(data):,}; còn thiếu: {data['area_was_missing'].sum():,}.")
    return data


def normalize_dates_and_phones(data):
    """Chuẩn hóa ngày đăng và số liên hệ."""
    posting_date = [
        parse_date(listing["posted_date"], listing.get("crawled_at", ""))
        for _, listing in data.iterrows()
    ]
    data["posted_date"] = pd.to_datetime(posting_date, utc=True).tz_convert("Asia/Ho_Chi_Minh")
    previous_count = len(data)
    data = data.loc[data["posted_date"].notna()].copy()
    print(f"Loại ngày chưa xác định (không đoán ngày hiện tại): {previous_count - len(data):,} tin.")
    data["phone"] = data["phone"].str.replace(r"[ .()\-]", "", regex=True)
    data["phone"] = data["phone"].str.replace(r"^[Oo](?=\d)", "0", regex=True)
    data["phone"] = data["phone"].str.extract(r"(?<!\d)(\+?84\d{9}|0\d{9,10})(?!\d)", expand=False).fillna("")
    data["phone"] = data["phone"].str.replace(r"^\+?84", "0", regex=True)
    data["phone"] = data["phone"].where(data["phone"].str.fullmatch(r"0\d{9}|0[12]\d{9}"), "")
    for index in data.index[data["phone"].eq("")]:
        text = "\n".join(data.loc[index, ["description", "address", "title"]])
        phone_number = re.search(r"(?<!\d)(?:\+?84|0)(?:[ .()\-]*\d){9,10}(?![ .()\-]*\d)", text)
        if phone_number:
            phone_number = re.sub(r"[ .()\-]", "", phone_number.group())
            phone_number = re.sub(r"^\+?84", "0", phone_number)
            if re.fullmatch(r"0\d{9}|0[12]\d{9}", phone_number):
                data.at[index, "phone"] = phone_number
                data.at[index, "recovery_note"] += "phone:source_text;"
    return data


def reconcile_locations(raw, data):
    """Đối chiếu địa danh và loại các tin mâu thuẫn vị trí."""
    # Nhãn quận/huyện lịch sử dùng trong dữ liệu, không phải cấp hành chính hiện hành.
    # Phường cũ trùng nhiều quận hoặc trùng tên phường mới đã được bỏ khỏi bảng này.
    # Bảng phường cũ: snapshot dvhcvn 01/03/2025, đã bỏ tên mơ hồ.
    # Nguồn: github.com/daohoangson/dvhcvn, commit eaddcea82fc336d59e908989f04178c336a9db10.
    # Bảng được nhúng tại đây; khi chạy không đọc model/CSV dự án cũ hoặc gọi mạng.
    WARDS_BY_DISTRICT = {
        'Quận Ba Đình': 'Cống Vị|Kim Mã|Liễu Giai|Ngọc Khánh|Phúc Xá|Quán Thánh|Thành Công|Trúc Bạch|Vĩnh Phúc|Điện Biên|Đội Cấn',
        'Quận Hoàn Kiếm': 'Cửa Đông|Hàng Buồm|Hàng Bài|Hàng Bông|Hàng Bạc|Hàng Bồ|Hàng Gai|Hàng Mã|Hàng Trống|Hàng Đào|Lý Thái Tổ|Phan Chu Trinh|Phúc Tân|Tràng Tiền|Trần Hưng Đạo',
        'Quận Tây Hồ': 'Bưởi|Nhật Tân|Quảng An|Thụy Khuê|Tứ Liên|Xuân La|Yên Phụ',
        'Quận Long Biên': 'Cự Khối|Gia Thụy|Giang Biên|Ngọc Lâm|Ngọc Thụy|Phúc Đồng|Thượng Thanh|Thạch Bàn',
        'Quận Cầu Giấy': 'Dịch Vọng|Dịch Vọng Hậu|Mai Dịch|Nghĩa Tân|Quan Hoa',
        'Quận Đống Đa': 'Cát Linh|Hàng Bột|Khâm Thiên|Khương Thượng|Láng Hạ|Láng Thượng|Nam Đồng|Phương Liên - Trung Tự|Phương Mai|Thịnh Quang|Thổ Quan|Trung Liệt|Văn Chương',
        'Quận Hai Bà Trưng': 'Bách Khoa|Bạch Đằng|Lê Đại Hành|Nguyễn Du|Phạm Đình Hổ|Phố Huế|Quỳnh Mai|Thanh Lương|Thanh Nhàn|Trương Định|Đồng Nhân',
        'Quận Hoàng Mai': 'Giáp Bát|Mai Động|Thịnh Liệt|Tân Mai|Đại Kim',
        'Quận Thanh Xuân': 'Hạ Đình|Khương Mai|Khương Trung|Nhân Chính|Thanh Xuân Bắc|Thanh Xuân Trung|Thượng Đình',
        'Huyện Sóc Sơn': 'Bắc Phú|Bắc Sơn|Hiền Ninh|Hồng Kỳ|Kim Lũ|Mai Đình|Minh Phú|Minh Trí|Nam Sơn|Phù Linh|Phù Lỗ|Quang Tiến|Tiên Dược|Tân Hưng|Việt Long|Xuân Giang|Xuân Thu|Đức Hoà',
        'Huyện Đông Anh': 'Bắc Hồng|Cổ Loa|Dục Tú|Hải Bối|Kim Nỗ|Mai Lâm|Nam Hồng|Nguyên Khê|Thuỵ Lâm|Tiên Dương|Tàm Xá|Uy Nỗ|Vân Hà|Vân Nội|Võng La|Vĩnh Ngọc|Xuân Canh|Xuân Nộn|Đông Hội|Đại Mạch',
        'Huyện Gia Lâm': 'Cổ Bi|Dương Quang|Dương Xá|Kim Đức|Kiêu Kỵ|Lệ Chi|Ninh Hiệp|Thiên Đức|Trâu Quỳ|Yên Thường|Yên Viên|Đa Tốn|Đặng Xá',
        'Quận Nam Từ Liêm': 'Cầu Diễn|Mễ Trì|Mỹ Đình 1|Mỹ Đình 2|Phú Đô|Phương Canh|Trung Văn',
        'Huyện Thanh Trì': 'Duyên Hà|Hữu Hoà|Liên Ninh|Ngũ Hiệp|Tân Triều|Tả Thanh Oai|Tứ Hiệp|Văn Điển|Vĩnh Quỳnh|Yên Mỹ|Đông Mỹ|Đại áng',
        'Quận Bắc Từ Liêm': 'Cổ Nhuế 1|Cổ Nhuế 2|Phúc Diễn|Thụy Phương|Xuân Tảo|Đức Thắng',
        'Huyện Mê Linh': 'Chi Đông|Chu Phan|Hoàng Kim|Kim Hoa|Tam Đồng|Thanh Lâm|Thạch Đà|Tiến Thịnh|Tráng Việt|Tự Lập|Văn Khê|Đại Thịnh',
        'Quận Hà Đông': 'Biên Giang|Hà Cầu|La Khê|Mộ Lao|Phú La|Phú Lãm|Phúc La|Văn Quán|Đồng Mai',
        'Thị xã Sơn Tây': 'Cổ Đông|Kim Sơn|Ngô Quyền|Phú Thịnh|Sơn Lộc|Thanh Mỹ|Trung Hưng|Trung Sơn Trầm|Viên Sơn|Xuân Khanh|Xuân Sơn|Đường Lâm',
        'Huyện Ba Vì': 'Ba Trại|Cam Thượng|Chu Minh|Cẩm Lĩnh|Khánh Thượng|Minh Quang|Phong Vân|Phú Châu|Phú Hồng|Sơn Đà|Thuần Mỹ|Thụy An|Tây Đằng|Tòng Bạt|Tản Lĩnh|Vân Hòa|Vạn Thắng|Đồng Thái',
        'Huyện Phúc Thọ': 'Hiệp Thuận|Liên Hiệp|Long Thượng|Nam Hà|Ngọc Tảo|Phúc Hòa|Phụng Thượng|Sen Phương|Tam Thuấn|Thanh Đa|Trạch Mỹ Lộc|Tích Lộc|Võng Xuyên',
        'Huyện Đan Phượng': 'Hạ Mỗ|Liên Hồng|Liên Trung|Phùng|Phương Đình|Thượng Mỗ|Thọ An|Thọ Xuân|Trung Châu|Tân Hội|Tân Lập|Đồng Tháp',
        'Huyện Hoài Đức': 'An Thượng|Cát Quế|Di Trạch|Dương Liễu|La Phù|Lại Yên|Tiền Yên|Trạm Trôi|Vân Canh|Vân Côn|Đông La|Đắc Sở|Đức Thượng',
        'Huyện Quốc Oai': 'Cấn Hữu|Cộng Hòa|Hòa Thạch|Liệp Nghĩa|Ngọc Liệp|Ngọc Mỹ|Phú Mãn|Phượng Sơn|Sài Sơn|Thạch Thán|Tuyết Nghĩa|Đông Yên',
        'Huyện Thạch Thất': 'Bình Yên|Cần Kiệm|Cẩm Yên|Hương Ngải|Kim Quan|Lam Sơn|Liên Quan|Lại Thượng|Phú Kim|Thạch Hoà|Thạch Xá|Tiến Xuân|Tân Xã|Yên Bình|Yên Trung|Đại Đồng|Đồng Trúc',
        'Huyện Chương Mỹ': 'Chúc Sơn|Hoàng Diệu|Hồng Phú|Hợp Đồng|Hữu Văn|Lam Điền|Mỹ Lương|Nam Phương Tiến|Ngọc Hòa|Phụng Châu|Thanh Bình|Thượng Vực|Thụy Hương|Thủy Xuân Tiên|Tiên Phương|Trường Yên|Tân Tiến|Tốt Động|Văn Võ|Đông Phương Yên|Đông Sơn|Đại Yên|Đồng Lạc',
        'Huyện Thanh Oai': 'Bích Hòa|Cao Viên|Cao Xuân Dương|Cự Khê|Hồng Dương|Kim An|Kim Bài|Kim Thư|Liên Châu|Mỹ Hưng|Phương Trung|Thanh Cao|Thanh Mai|Thanh Thùy|Thanh Văn|Tân Ước|Đỗ Động',
        'Huyện Thường Tín': 'Duyên Thái|Dũng Tiến|Hiền Giang|Hà Hồi|Hòa Bình|Khánh Hà|Liên Phương|Lê Lợi|Minh Cường|Nghiêm Xuyên|Nguyễn Trãi|Nhị Khê|Ninh Sở|Quất Động|Thắng Lợi|Tô Hiệu|Tự Nhiên|Vân Tảo|Văn Bình|Văn Phú|Vạn Nhất',
        'Huyện Phú Xuyên': 'Bạch Hạ|Châu Can|Hoàng Long|Hồng Minh|Hồng Thái|Khai Thái|Minh Tân|Nam Phong|Nam Tiến|Phú Túc|Phú Yên|Phúc Tiến|Quang Hà|Quang Lãng|Tri Thủy|Văn Hoàng',
        'Huyện Ứng Hòa': 'Bình Lưu Quang|Cao Sơn Tiến|Hoa Viên|Kim Đường|Liên Bạt|Minh Đức|Phù Lưu|Phương Tú|Quảng Phú Cầu|Trung Tú|Trường Thịnh|Trầm Lộng|Tảo Dương Văn|Đông Lỗ|Đại Cường|Đồng Tân',
        'Huyện Mỹ Đức': 'An Mỹ|An Phú|An Tiến|Hùng Tiến|Hợp Thanh|Hợp Tiến|Lê Thanh|Mỹ Xuyên|Phù Lưu Tế|Phúc Lâm|Thượng Lâm|Tuy Lai|Vạn Tín|Xuy Xá|Đại Nghĩa',
    }
    district_by_name = {}
    district_by_ward = {}
    district_patterns = {}
    for district, wards in WARDS_BY_DISTRICT.items():
        name = strip_accents(district)
        short_name = re.sub(r"^(?:quan|huyen|thi xa)\s+", "", name)
        district_by_name.update({name: district, short_name: district})
        if district.startswith("Quận"):
            prefix = r"(?:quan\s+|q[.\s]*)"
        elif district.startswith("Huyện"):
            prefix = r"(?:huyen\s+|h[.\s]*)"
        else:
            prefix = r"(?:thi xa\s+|tx[.\s]*)"
        district_patterns[district] = re.compile(r"\b" + prefix + re.escape(short_name) + r"\b")
        for ward in wards.split("|"):
            if ward:
                district_by_ward[strip_accents(ward)] = district
    # Từ vựng lấy từ chính địa chỉ raw: chỉ dùng để nhận diện tên, không tự gán quận.
    canonical_streets, canonical_wards = {}, {}
    for address in raw["address"]:
        for hit in re.finditer(r"(?:^|[,;])\s*(?:phường|xã|thị trấn|p\.|x\.)\s*([^,;]+)", address, re.I):
            name = re.split(r"\s+(?:quận|huyện|thành phố)\b", hit[1], flags=re.I)[0].strip()
            canonical_wards[strip_accents(name)] = name
        for hit in re.finditer(r"\b(?:đường|phố|duong|pho)\s+([^,;]+)", address, re.I):
            name = re.split(r"\s*-\s*(?=[^\W\d_])|\s+(?:phường|xã|quận|huyện|ngõ|ngách|phuong|xa|quan|huyen)\b", hit[1], flags=re.I)[0].strip()
            if 2 <= len(name.split()) <= 6 and len(name) <= 45 and not re.search(r"\b(?:gan|cach|so|ngo|ngach|ha noi|hn|quan|huyen|phuong|thanh pho)\b", strip_accents(name)):
                canonical_streets[strip_accents(name)] = name
    street_pattern = r"\b(?:" + "|".join(re.escape(t) for t in sorted(canonical_streets, key=len, reverse=True)) + r")\b"

    data["_district_method_raw"] = data.get("district_method", "")
    for column in ["Đường", "Phường/Xã", "Quận/Huyện", "district_method"]:
        data[column] = "Chua_ro"
    data["_mau_thuan"] = False
    data["_ngoai_hanoi"] = False
    for index, listing in data.iterrows():
        sections = [listing["address"], listing["title"]]
        for row in listing["description"].splitlines():
            if re.search(r"\b(?:dia chi|d/c|vi tri|toa lac|tai)\b", strip_accents(row)):
                sections.append(row)
        address = listing["address"]
        # Chỉ lấy dòng tự nêu địa chỉ; bỏ câu quảng cáo "gần/cách ...".
        if not re.search(r"\d|\b(?:duong|pho|ngo|ngach)\b", strip_accents(address)):
            for row in listing["description"].splitlines():
                if re.match(r"\s*(?:[-*•]\s*)?(?:dia chi|d/c|vi tri|toa lac|tai)\b", strip_accents(row)) and re.search(r"\b(?:duong|pho|ngo|ngach)\b", strip_accents(row)):
                    address = re.sub(r"^\s*(?:[-*•]\s*)?(?:địa chỉ|d/c|vị trí|tọa lạc|tại)\s*[:：]?\s*", "", row, flags=re.I) + ", " + address
                    data.at[index, "address"] = address
                    data.at[index, "recovery_note"] += "address:description;"
                    break
        parts = [part.strip() for part in address.split(",") if part.strip()]
        ward = re.search(r"(?:^|[,;])\s*(?:phường|xã|thị trấn|p\.|x\.)\s*([^,;]+)", address, re.I)
        if ward:
            ward_name = re.split(r"\s+(?:quận|huyện|thành phố)\b", ward.group(1), flags=re.I)[0].strip()
        elif len(parts) >= 3:
            ward_name = parts[-2]
            if strip_accents(ward_name) not in canonical_wards or strip_accents(ward_name) in district_by_name or re.match(r"^(?:duong|pho|ngo|so)\b", strip_accents(ward_name)):
                ward_name = ""
        else:
            ward_name = ""
        if not ward_name:
            # Địa chỉ ngăn bằng dấu gạch ngang: thôn/đường - phường - quận - Hà Nội.
            components = [x.strip() for x in re.split(r"[,;]|\s*-\s*", address) if x.strip()]
            if len(components) >= 3:
                for component in reversed(components[1:]):
                    name = strip_accents(component).strip(" .")
                    if name in canonical_wards and name not in district_by_name:
                        ward_name = canonical_wards[name]
                        data.at[index, "recovery_note"] += "ward:address_component;"
                        break

        if not ward_name:
            wards = []
            for paragraph in sections:
                for hit in re.finditer(r"\b(?:phường|xã|thị trấn|p\.|x\.)\s+([^,;.\n]+)", paragraph, re.I):
                    name = re.split(r"\s+(?:quận|huyện|thành phố|hà nội)\b", hit[1], flags=re.I)[0].strip()
                    if strip_accents(name) in canonical_wards:
                        wards.append(canonical_wards[strip_accents(name)])
            if len(set(wards)) == 1:
                ward_name = wards[0]
                data.at[index, "recovery_note"] += "ward:address_text;"

        if not ward and listing["source"] == "phongtro.vn":
            hit = re.search(r"-(?:phuong|xa|thi-tran)-(.+?)-tp-ha-noi/", listing["url"])
            if hit:
                name = hit[1].replace("-", " ")
                ward_name = canonical_wards.get(name, name.title())
                data.at[index, "recovery_note"] += "ward:url;"

        street = re.search(r"\b(?:đường|phố|duong|pho)\s+([^,;]+)", address, re.I)
        if street:
            street_name = re.split(r"\s*-\s*(?=[^\W\d_])|\s+(?:phường|xã|quận|huyện|phuong|xa|quan|huyen)\b", street.group(1), flags=re.I)[0].strip()
            data.at[index, "Đường"] = street_name
        if data.at[index, "Đường"] == "Chua_ro":
            # Với địa chỉ có cấu trúc, bỏ thành phần quận/phường ở cuối trước khi tìm đường.
            street_parts = [part for i, part in enumerate(parts) if i == 0 or (strip_accents(part) not in district_by_name and strip_accents(part) not in ("ha noi", strip_accents(ward_name)))]
            street_section = ", ".join(street_parts)
            search_sections = [street_section]
            if not re.search(r"\d|\b(?:duong|pho|ngo|ngach)\b", strip_accents(address)):
                search_sections += sections[2:]
            for paragraph in search_sections:
                text = strip_accents(paragraph)
                # Cần số nhà/ngõ hoặc cấu trúc địa chỉ, không đoán đường từ tên quận/phường.
                if not re.search(r"\d|\b(?:duong|pho|ngo|ngach)\b", text) and not (paragraph == street_section and len(parts) >= 3):
                    continue
                candidate_names = []
                for hit in re.finditer(street_pattern, text):
                    preceding_text = text[max(0, hit.start()-45):hit.start()]
                    if hit[0] in district_by_name and not re.search(r"\b(?:duong|pho|ngo|ngach)\b[^,;]{0,20}$", preceding_text):
                        continue
                    if not re.search(r"\b(?:phuong|xa|quan|huyen|khu|kdt|thon|xom|gan|cach|giap|sat|doi dien)\b[^,;]{0,25}$", preceding_text):
                        candidate_names.append(hit[0])
                if len(set(candidate_names)) == 1:
                    data.at[index, "Đường"] = canonical_streets[candidate_names[0]]
                    data.at[index, "recovery_note"] += "street:address_text;"
                    break
        if ward_name:
            ward_name = canonical_wards.get(strip_accents(ward_name), ward_name)
            data.at[index, "Phường/Xã"] = ward_name
        candidates = set()
        metadata = district_by_name.get(strip_accents(listing.get("district_raw", "")))
        if metadata:
            candidates.add(metadata)
        for paragraph in sections:
            text = strip_accents(paragraph)
            if re.search(r"\b(?:ho chi minh|tphcm|tp\.?\s*hcm|thu duc|go vap|binh thanh|phu nhuan|da nang|(?:quan\s+|q[.\s]*)(?:1[0-2]|[1-9]|tan binh|tan phu|binh tan))\b", text):
                data.at[index, "_ngoai_hanoi"] = True
            for district, pattern in district_patterns.items():
                for hit in pattern.finditer(text):
                    preceding_text = text[max(0, hit.start() - 45):hit.start()]
                    if not re.search(r"\b(?:gan|cach|giap|doi dien|huong ve)\b[^,;]{0,30}$", preceding_text):
                        candidates.add(district)
        if len(parts) >= 4 and ward:
            trailing_district = district_by_name.get(strip_accents(parts[-2]))
            if trailing_district:
                candidates.add(trailing_district)
        district_from_ward = district_by_ward.get(strip_accents(ward_name))
        if district_from_ward:
            candidates.add(district_from_ward)
        if len(candidates) > 1:
            data.at[index, "_mau_thuan"] = True
        elif len(candidates) == 1:
            data.at[index, "Quận/Huyện"] = candidates.pop()
            data.at[index, "district_method"] = (listing.get("_district_method_raw", "") or "metadata") if metadata else "text_or_ward"
    previous_count = len(data)
    print(f"Mâu thuẫn quận: {data['_mau_thuan'].sum():,}; dấu hiệu ngoài Hà Nội: {data['_ngoai_hanoi'].sum():,}.")
    data = data.loc[~data["_mau_thuan"] & ~data["_ngoai_hanoi"]].copy()
    print(f"Loại vị trí mâu thuẫn/sai vùng: {previous_count - len(data):,} tin.")
    return data, district_by_name


def deduplicate_and_cross_check(data):
    """Bỏ tin trùng và đối chiếu quận theo đường, phường."""
    data["_do_dai"] = data["description"].str.len()
    data = data.sort_values(["posted_date", "_do_dai"], kind="stable")
    previous_count = len(data)
    for column in ["post_id", "url"]:
        has_key = data[column].fillna("").str.strip().ne("")
        is_duplicate = has_key & data.duplicated(["source", column], keep="last")
        data = data.loc[~is_duplicate].copy()
    data["_title"] = data["title"].map(strip_accents)
    data["_desc"] = data["description"].map(strip_accents)
    data["_address"] = data["address"].map(strip_accents)
    has_long_text = data["_title"].str.len().ge(20) & data["_desc"].str.len().ge(80)
    is_duplicate = has_long_text & data.duplicated(["_title", "_desc", "price", "area"], keep="last")
    data = data.loc[~is_duplicate].copy()
    # Cùng liên hệ + địa chỉ cụ thể + mô tả/giá/diện tích giống hệt, kể cả đổi tiêu đề.
    has_contact = data["phone"].ne("") & data["_address"].str.contains(r"\d") & data["_desc"].str.len().ge(80)
    is_duplicate = has_contact & data.duplicated(["phone", "_address", "_desc", "price", "area"], keep="last")
    data = data.loc[~is_duplicate].copy()
    print(f"Xóa tin trùng URL/mã/nội dung: {previous_count - len(data):,} tin.")

    # Đối chiếu bổ sung: mỗi cặp đường|phường có >=5 URL độc lập, breadcrumb cùng một quận.
    # Snapshot crawl 02/11/2026; không dùng giá để lập bảng, không ép quận chỉ từ phường mới.
    STREET_WARD_REFERENCE = {
        'Huyện Hoài Đức': 'lai xa|hoai duc;quoc lo 32|hoai duc;quoc lo 32|thuan an;so 70|son dong;thu y|hoai duc',
        'Huyện Thanh Trì': 'ngoc hoi|thanh tri',
        'Huyện Đông Anh': 'ngoc chi|vinh thanh',
        'Quận Ba Đình': 'buoi|ngoc ha;dao tan|ngoc ha;de la thanh|giang vo;doi can|ba dinh;doi can|ngoc ha;hoang hoa tham|ba dinh;hoang hoa tham|ngoc ha;kim ma|giang vo;la thanh|giang vo;nghia dung|hong ha;ngoc ha|ba dinh;ngoc khanh|giang vo;nguyen hong|giang vo;nui truc|giang vo;tan ap|hong ha;van cao|ngoc ha',
        'Quận Bắc Từ Liêm': 'cau dien|phu dien;cau dien|tay tuu;cau noi|dong ngac;co nhue|dong ngac;duc dien|phu dien;hoang tang bi|dong ngac;nguyen dao an|phu dien;pham van dong|dong ngac;pham van dong|nghia do;pham van dong|xuan dinh;pham van nghi|dong ngac;phu dien|phu dien;phu kieu|phu dien;so 32|phu dien;van tri|tay tuu;xuan dinh|xuan dinh',
        'Quận Cầu Giấy': 'bo song|nghia do;cau giay|cau giay;cau giay|nghia do;dich vong hau|cau giay;dich vong|cau giay;doan ke thien|phu dien;duong quang ham|nghia do;duy tan|cau giay;ho tung mau|cau giay;hoa bang|cau giay;lac long quan|nghia do;mai dich|phu dien;nguyen dinh hoan|nghia do;nguyen khang|cau giay;nguyen khanh toan|nghia do;pham tuan tai|cau giay;pham van dong|phu dien;quan hoa|nghia do;to hieu|nghia do;tran binh|phu dien;tran duy hung|cau giay;tran quoc hoan|cau giay;tran quoc vuong|cau giay;tran thai tong|cau giay;trung kinh|cau giay;vo chi cong|nghia do;xuan thuy|cau giay;yen hoa|cau giay',
        'Quận Hai Bà Trưng': 'bach mai|bach mai;dai la|bach mai;dai la|phu dong;duong van be|vinh tuy;giai phong|bach mai;goc de|bach mai;hoa binh 7|bach mai;hong mai|bach mai;kim nguu|bach mai;kim nguu|vinh tuy;lac trung|vinh tuy;le thanh nghi|bach mai;minh khai|bach mai;nguyen khoai|vinh tuy;pho vong|bach mai;ta quang buu|bach mai;thanh nhan|bach mai;tran dai nghia|bach mai;tran khat chan|bach mai;tran khat chan|vinh tuy;truong dinh|phu dong;vinh tuy|vinh tuy',
        'Quận Hoàn Kiếm': 'cau dat|hong ha;phuc tan|hong ha',
        'Quận Hoàng Mai': 'bang a|hoang liet;cau lu|dinh cong;dai tu|dinh cong;dinh cong thuong|dinh cong;dinh cong|dinh cong;giai phong|hoang liet;giai phong|tuong mai;giap bat|tuong mai;hoang mai|tuong mai;kim dong|tuong mai;kim giang|dinh cong;kim giang|vinh hung;le trong tan|dinh cong;linh duong|hoang liet;linh nam|bat trang;linh nam|vinh hung;linh nam|vinh tuy;minh khai|tuong mai;nam du|bat trang;nguyen an ninh|thuong cat;nguyen duc canh|thuong cat;nguyen khoai|vinh hung;tam trinh|tuong mai;tam trinh|yen so;tran dien|dinh cong;truong dinh|thuong cat;vinh hung|vinh hung',
        'Quận Hà Đông': 'ba la|phu luong;ba trieu|ha dong;bach thai buoi|ha dong;chien thang|ha dong;da si|kien hung;ha tri|ha dong;kien hung|kien hung;le trong tan|duong noi;le trong tan|ha dong;mau luong|kien hung;mo lao|ha dong;ngo quyen|ha dong;nguyen khuyen|ha dong;nguyen van troi|ha dong;phan dinh giot|ha dong;phung hung|ha dong;quang trung|ha dong;so 19/5|ha dong;thanh binh|ha dong;to huu|ha dong;tran phu|ha dong;van phuc|ha dong;xa la|ha dong;xa la|kien hung;yen binh|ha dong;yen phuc|ha dong;yen xa|ha dong',
        'Quận Long Biên': 'duc giang|viet hung;ngo gia tu|viet hung;ngoc thuy|hong ha;nguyen van cu|dong ngac',
        'Quận Thanh Xuân': 'bui xuong trach|khuong dinh;chinh kinh|thanh xuan;cu chinh lan|phuong liet;giai phong|phuong liet;giap nhat|thanh xuan;ha dinh|khuong dinh;hoang ngan|thanh xuan;hoang van thai|khuong dinh;hoang van thai|phuong liet;khuat duy tien|thanh xuan;khuong dinh|khuong dinh;khuong ha|khuong dinh;khuong trung|khuong dinh;kim giang|khuong dinh;le trong tan|phuong liet;luong the vinh|thanh xuan;nguyen huy tuong|thanh xuan;nguyen lan|phuong liet;nguyen ngoc nai|phuong liet;nguyen trai|khuong dinh;nguyen trai|thanh xuan;nguyen xien|khuong dinh;nguyen xien|thanh xuan;phan dinh giot|phuong liet;phung khoang|thanh xuan;phuong liet|phuong liet;quan nhan|thanh xuan;trieu khuc|thanh xuan;truong chinh|phuong liet;vu tong phan|khuong dinh;vuong thua vu|khuong dinh',
        'Quận Tây Hồ': 'an duong vuong|phu thuong;an duong|hong ha;au co|hong ha;au co|tay ho;dang thai mai|tay ho;lac long quan|hong ha;lac long quan|tay ho;nghi tam|hong ha;thuy khue|tay ho;tu lien|hong ha;vo chi cong|tay ho;vong thi|tay ho;xuan la|tay ho',
        'Quận Đống Đa': 'cho kham thien|van mieu - quoc tu giam;chua lang|lang;dang tien dong|dong da;dang van ngu|kim lien;de la thanh|o cho dua;dong tac|kim lien;duong lang|lang;hao nam|o cho dua;hoang cau|o cho dua;kham thien|van mieu - quoc tu giam;khuong thuong|kim lien;kim hoa|kim lien;lang ha|lang;lang|dong da;lang|lang;luong dinh cua|kim lien;mai anh tuan|o cho dua;nguyen chi thanh|lang;nguyen khuyen|van mieu - quoc tu giam;nguyen luong bang|dong da;nguyen luong bang|kim lien;pham ngoc thach|kim lien;phao dai lang|lang;phuong mai|kim lien;tay son|dong da;tay son|kim lien;thai ha|dong da;thai thinh|dong da;thinh quang|dong da;ton duc thang|van mieu - quoc tu giam;ton that tung|kim lien;truong chinh|kim lien;xa dan|dong da;xa dan|kim lien;yen lang|dong da',
    }
    reference_table = {}
    for district, matched_pairs in STREET_WARD_REFERENCE.items():
        for segment in matched_pairs.split(";"):
            street, ward = segment.split("|")
            reference_table[(street, ward)] = district

    # Cùng đường + phường đã có >=2 tin độc lập và chỉ một quận: đối chiếu bổ sung.
    known_locations = data.loc[data["Quận/Huyện"].ne("Chua_ro") & data["Đường"].ne("Chua_ro") & data["Phường/Xã"].ne("Chua_ro")].copy()
    known_locations["_duong"] = known_locations["Đường"].map(strip_accents)
    known_locations["_phuong"] = known_locations["Phường/Xã"].map(strip_accents)
    district_table = known_locations.groupby(["_duong", "_phuong"])["Quận/Huyện"].agg(so_quan="nunique", so_tin="size", quan="first")
    # Nếu dữ liệu hiện có cho thấy cặp này thuộc nhiều quận, không dùng tham chiếu để ép chọn.
    for segment in district_table.index[district_table["so_quan"].gt(1)]:
        reference_table.pop(segment, None)
    district_table = district_table.loc[district_table["so_quan"].eq(1) & district_table["so_tin"].ge(2)]
    lookup = reference_table
    lookup.update(district_table["quan"].to_dict())
    for index in data.index[data["Quận/Huyện"].eq("Chua_ro")]:
        lookup_key = (strip_accents(data.at[index, "Đường"]), strip_accents(data.at[index, "Phường/Xã"]))
        if lookup_key in lookup:
            data.at[index, "Quận/Huyện"] = lookup[lookup_key]
            data.at[index, "district_method"] = "street_ward_match"
    return data


def recover_districts_from_address(data, district_by_name):
    """Bổ sung quận bằng địa danh và địa chỉ cụ thể."""
    # Danh mục địa danh Hà Nội, Thông tư 10/2022/TT-BTNMT; chỉ nhúng tên xuất hiện trong dữ liệu.
    # Giữ đủ các quận của đường liên quận, không lấy tùy tiện quận đầu tiên.
    # Nguồn: https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/10/10-btnmt.pdf
    OFFICIAL_STREET_DISTRICTS = {
        'Huyện Ba Vì': 'Quốc lộ 32',
        'Huyện Gia Lâm': 'Nguyễn Văn Linh|Nguyễn Đức Thuận|Quốc lộ 3|Thiên Đức',
        'Huyện Hoài Đức': 'Lê Trọng Tấn|Phương Canh|Quốc lộ 32|Xuân Phương|đại lộ Thăng Long',
        'Huyện Mê Linh': '100',
        'Huyện Phúc Thọ': 'Quốc lộ 32',
        'Huyện Quốc Oai': 'đại lộ Thăng Long',
        'Huyện Sóc Sơn': 'Quốc lộ 3',
        'Huyện Thanh Trì': 'Chiến Thắng|Cầu Bươu|Kim Giang|Nguyễn Xiển|Ngũ Hiệp|Ngọc Hồi|Phan Trọng Tuệ|Phùng Hưng|Triều Khúc|Vành Đai 3|Đông Mỹ',
        'Huyện Thạch Thất': 'Quốc lộ 32|đại lộ Thăng Long',
        'Huyện Đan Phượng': 'Phan Đình Phùng|Quốc lộ 32|Tây Sơn',
        'Huyện Đông Anh': '100|Quốc lộ 3',
        'Quận Ba Đình': 'Bưởi|Chu Văn An|Giảng Võ|Hoàng Hoa Thám|Huỳnh Thúc Kháng|Hồng Hà|Kim Mã|La Thành|Liễu Giai|Láng Hạ|Lê Hồng Phong|Nguyên Hồng|Nguyễn Chí Thanh|Nguyễn Công Hoan|Nguyễn Trường Tộ|Ngọc Hà|Ngọc Khánh|Núi Trúc|Phan Đình Phùng|Quán Thánh|Thái Hà|Trần Phú|Văn Cao|Vạn Bảo|Vạn Phúc|Đào Tấn|Độc Lập|Đội Cấn',
        'Quận Bắc Từ Liêm': 'An Dương Vương|Cầu Diễn|Cổ Nhuế|Hoàng Công Chất|Hoàng Quốc Việt|Hoàng Tăng Bí|Hồ Tùng Mậu|Liên Mạc|Nguyễn Hoàng Tôn|Phan Bá Vành|Phú Diễn|Phúc Diễn|Phạm Văn Đồng|Quốc lộ 32|Thuỵ Phương|Thượng Cát|Trần Cung|Tân Phong|Tây Tựu|Viên|Văn Tiến Dũng|Xuân La|Xuân Phương|Xuân Đỉnh|Đông Ngạc|Đỗ Nhuận|Đức Thắng',
        'Quận Cầu Giấy': 'Bưởi|Cầu Giấy|Doãn Kế Thiện|Duy Tân|Dương Khuê|Dương Quảng Hàm|Dương Đình Nghệ|Dịch Vọng|Dịch Vọng Hậu|Hoàng Ngân|Hoàng Quốc Việt|Hoàng Sâm|Hồ Tùng Mậu|Khuất Duy Tiến|Lê Văn Lương|Lê Đức Thọ|Lạc Long Quân|Nghĩa Tân|Nghĩa Đô|Nguyễn Chánh|Nguyễn Khang|Nguyễn Khánh Toàn|Nguyễn Khả Trạc|Nguyễn Ngọc Vũ|Nguyễn Phong Sắc|Nguyễn Thị Định|Nguyễn Văn Huyên|Nguyễn Đình Hoàn|Phùng Chí Kiên|Phạm Hùng|Phạm Văn Đồng|Quan Hoa|Quan Nhân|Quốc lộ 32|Thành Thái|Trung Hoà|Trung Kính|Trương Công Giai|Trần Bình|Trần Cung|Trần Duy Hưng|Trần Quốc Hoàn|Trần Quốc Vượng|Trần Thái Tông|Trần Đăng Ninh|Tô Hiệu|Võ Chí Công|Xuân Thuỷ|Yên Hoà|Đỗ Quang|đại lộ Thăng Long',
        'Quận Hai Bà Trưng': 'Bà Triệu|Bạch Mai|Bạch Đằng|Giải Phóng|Hoàng Mai|Huế|Kim Ngưu|Lê Quý Đôn|Lê Văn Hưu|Lê Đại Hành|Minh Khai|Nguyễn An Ninh|Nguyễn Du|Ngô Thì Nhậm|Quang Trung|Thanh Nhàn|Trương Định|Trần Hưng Đạo|Trần Khát Chân|Trần Nhân Tông|Trần Đại Nghĩa|Tuệ Tĩnh|Tương Mai|Vân Đồn|Võ Thị Sáu|Vĩnh Tuy|Vọng|Yên Bái 1|Đại La',
        'Quận Hoàn Kiếm': 'Bà Triệu|Bạch Đằng|Cầu Gỗ|Cầu Đông|Hai Bà Trưng|Huế|Hàng Bông|Hàng Tre|Hồng Hà|Ngô Thì Nhậm|Phan Đình Phùng|Phùng Hưng|Phúc Tân|Quang Trung|Quán Sứ|Thanh Hà|Tràng Thi|Tràng Tiền|Trần Hưng Đạo|Trần Phú|Trần Quốc Toản|Vọng Hà|Đường Thành',
        'Quận Hoàng Mai': 'Bằng Liệt|Giáp Bát|Giải Phóng|Hoàng Liệt|Hoàng Mai|Kim Giang|Kim Đồng|Linh Đường|Lĩnh Nam|Lương Khánh Thiện|Mai Động|Nguyễn An Ninh|Nguyễn Chính|Nguyễn Xiển|Nguyễn Đức Cảnh|Ngọc Hồi|Trương Định|Tân Mai|Vĩnh Hưng|Vũ Tông Phan|Định Công|Định Công Hạ',
        'Quận Hà Đông': 'Ba La|Bà Triệu|Chiến Thắng|Chu Văn An|Cầu Bươu|Hữu Hưng|La Dương|La Nội|Lê Hồng Phong|Lê Trọng Tấn|Nguyễn Khuyến|Nguyễn Trãi|Nguyễn Trực|Nguyễn Viết Xuân|Nguyễn Văn Lộc|Nguyễn Văn Trỗi|Ngô Thì Nhậm|Phan Đình Giót|Phùng Hưng|Phú Lương|Quang Trung|Thanh Bình|Trần Phú|Tô Hiệu|Tố Hữu|Văn La|Văn Phú|Vạn Phúc|Xốm|Đa Sỹ',
        'Quận Long Biên': 'Bát Khối|Bồ Đề|Cổ Linh|Lý Sơn|Nguyễn Sơn|Nguyễn Văn Cừ|Nguyễn Văn Linh|Nguyễn Đức Thuận|Ngô Gia Tự|Ngọc Lâm|Ngọc Thuỵ|Phúc Lợi|Sài Đồng|Thạch Bàn|Việt Hưng|Vũ Xuân Thiều|Đức Giang',
        'Quận Nam Từ Liêm': 'Châu Văn Liêm|Duy Tân|Dương Khuê|Dương Đình Nghệ|Hàm Nghi|Hồ Tùng Mậu|Hữu Hưng|Lê Quang Đạo|Lê Đức Thọ|Lương Thế Vinh|Miếu Đầm|Mễ Trì|Mễ Trì Thượng|Mỹ Đình|Nguyễn Cơ Thạch|Nguyễn Hoàng|Nguyễn Trãi|Ngọc Trục|Nhổn|Phúc Diễn|Phương Canh|Phạm Hùng|Quang Tiến|Quốc lộ 32|Sa Đôi|Trung Văn|Trần Bình|Tân Mỹ|Tây Mỗ|Tố Hữu|Xuân Phương|Đình Thôn|Đại Linh|Đại Mỗ|Đồng Me|Đỗ Đức Dục|đại lộ Thăng Long',
        'Quận Thanh Xuân': 'Chính Kinh|Giải Phóng|Hoàng Ngân|Hoàng Văn Thái|Khuất Duy Tiến|Khương Hạ|Khương Trung|Khương Đình|Kim Giang|Lê Trọng Tấn|Lê Văn Lương|Lương Thế Vinh|Nguyễn Huy Tưởng|Nguyễn Lân|Nguyễn Thị Định|Nguyễn Trãi|Nguyễn Viết Xuân|Nguyễn Xiển|Nguỵ Như Kon Tum|Nhân Hoà|Quan Nhân|Thượng Đình|Triều Khúc|Trường Chinh|Tố Hữu|Vũ Trọng Phụng|Vũ Tông Phan|Vương Thừa Vũ|Vọng|Định Công',
        'Quận Tây Hồ': 'An Dương Vương|Hoàng Hoa Thám|Hồng Hà|Lạc Long Quân|Nguyễn Hoàng Tôn|Nguyễn Đình Thi|Phú Gia|Phú Thượng|Phú Xá|Thuỵ Khuê|Tô Ngọc Vân|Võ Chí Công|Võng Thị|Văn Cao|Xuân Diệu|Xuân La|Đặng Thai Mai',
        'Quận Đống Đa': 'Chùa Bộc|Chùa Láng|Cát Linh|Giải Phóng|Giảng Võ|Hoàng Cầu|Huỳnh Thúc Kháng|Hào Nam|Khâm Thiên|La Thành|Láng|Láng Hạ|Nguyên Hồng|Nguyễn Chí Thanh|Nguyễn Lương Bằng|Nguyễn Trãi|Pháo Đài Láng|Phạm Ngọc Thạch|Quốc Tử Giám|Thái Hà|Thái Thịnh|Trường Chinh|Tây Sơn|Tôn Đức Thắng|Vọng|Xã Đàn|Yên Lãng|Ô Chợ Dừa|Đê La Thành|Đặng Tiến Đông',
        'Thị xã Sơn Tây': 'La Thành|Quốc lộ 32',
    }
    districts_by_street, official_street_names = {}, {}
    for district, streets in OFFICIAL_STREET_DISTRICTS.items():
        for name in streets.split("|"):
            lookup_key = strip_accents(name)
            districts_by_street.setdefault(lookup_key, set()).add(district)
            official_street_names[lookup_key] = name
    # Bổ sung địa danh có công bố riêng, không suy diễn từ tên phường mới.
    # Phú Mỹ: https://sovhtt.hanoi.gov.vn/le-gan-bien-ten-duong-phu-my-quan-nam-tu-liem/
    # Mễ Trì Hạ: https://sovhtt.hanoi.gov.vn/phuong-me-tri-thi-dau-the-thao-chao-mung-ky-niem-10-nam-thanh-lap-quan-nam-tu-liem/
    for name in ["Phú Mỹ", "Mễ Trì Hạ"]:
        lookup_key = strip_accents(name)
        districts_by_street.setdefault(lookup_key, set()).add("Quận Nam Từ Liêm")
        official_street_names[lookup_key] = name

    # Nhận tên đầy đủ trước: tránh nhầm "Lãng Yên" thành "Láng" khi bỏ dấu.
    full_street_names = set(official_street_names)
    for name in data["Đường"]:
        lookup_key = strip_accents(name)
        unprefixed_name = re.sub(r"^(?:duong|pho)\s+", "", lookup_key)
        if unprefixed_name in official_street_names:
            lookup_key = unprefixed_name
        if name != "Chua_ro" and 2 <= len(lookup_key.split()) <= 6 and not re.search(r"\b(?:quan|huyen|phuong|thi tran|ha noi|hn|gan|cach)\b", lookup_key):
            full_street_names.add(lookup_key)
    street_name_pattern = r"\b(?:" + "|".join(re.escape(x) for x in sorted(full_street_names, key=len, reverse=True)) + r")\b"
    previously_known = data["Quận/Huyện"].ne("Chua_ro")
    data["_duong_doi_soat"] = ""
    data["_so_nha"] = ""
    data["_ngo"] = ""
    data["_phuong_doi_soat"] = data["Phường/Xã"].map(strip_accents)
    directly_matched_district = {}
    for index, listing in data.iterrows():
        addresses = [listing["address"]]
        if listing.get("address_detail", ""):
            addresses.append(listing["address_detail"])
        for row in listing["description"].splitlines():
            if re.search(r"\b(?:dia chi|d/c|nha minh o|nha tai|phong tai|toa lac)\b", strip_accents(row)):
                addresses.append(row)
        districts = set()
        for address in addresses:
            original_text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", address)).strip().casefold().replace("–", "-").replace("—", "-")
            text = strip_accents(original_text)
            # Quận viết thiếu tiền tố: chỉ nhận thành phần địa chỉ; tránh nhầm phường Cầu Giấy thành quận.
            if not previously_known.at[index]:
                for component in re.split(r"[,;]|\s*-\s*", text):
                    component = component.strip(" .")
                    district = district_by_name.get(component)
                    ward_name = listing["_phuong_doi_soat"]
                    existing_street_name = strip_accents(listing["Đường"])
                    if district and component not in (ward_name, existing_street_name):
                        districts.add(district)
                # Ví dụ "Trung Văn Nam Từ Liêm Hà Nội" không có dấu phẩy.
                for name, district in district_by_name.items():
                    if name in (listing["_phuong_doi_soat"], strip_accents(listing["Đường"])):
                        continue
                    district_match = re.search(r"\b" + re.escape(name) + r"\s*[,.-]?\s*(?:tp\.?\s*)?(?:ha noi|hn)\b", text)
                    if district_match:
                        before_district = text[max(0, district_match.start()-40):district_match.start()]
                        if not re.search(r"\b(?:duong|pho|phuong|xa)\s*$|\b(?:gan|cach|giap|doi dien)\b[^,;]{0,30}$", before_district):
                            districts.add(district)
            matches = []
            for hit in re.finditer(street_name_pattern, text):
                if hit[0] not in official_street_names:
                    continue
                if hit[0] == "lang" and len(original_text) == len(text) and original_text[hit.start():hit.end()] in ("làng", "lãng"):
                    continue
                preceding_text = text[max(0, hit.start()-50):hit.start()]
                has_street_marker = re.search(r"\b(?:duong|pho|ngo|ngach|hem)\s*(?:\d+[a-z]?(?:/\d+)*\s*)?$", preceding_text)
                if re.search(r"\b(?:phuong|xa|quan|huyen|thi tran|thon|lang|to dan pho|kdt|khu do thi|gan|cach|canh|sat|giap|doi dien)\b[^,;]{0,35}$", preceding_text):
                    continue
                has_house_number = re.search(r"(?<!\w)\d+[a-z]?(?:/\d+[a-z]?)*\s*$", preceding_text)
                if len(hit[0].split()) == 1 or hit[0] in district_by_name or listing["_phuong_doi_soat"].startswith(hit[0]):
                    if not (has_street_marker or has_house_number):
                        continue
                matches.append(hit)
            if len(matches) != 1 or data.at[index, "_duong_doi_soat"]:
                continue
            hit = matches[0]
            data.at[index, "_duong_doi_soat"] = hit[0]
            # Tách chuỗi số thực sự đứng trước tên đường; không lấy giá, diện tích hay số điện thoại.
            preceding_text = re.sub(r"\b(?:duong|pho)\s*$", "", text[:hit.start()]).strip()
            alleys = re.findall(r"\b(?:ngo|ngach|hem)\s*(\d+[a-z]?(?:/\d+[a-z]?)*)", preceding_text)
            alley = ""
            for segment in reversed(alleys):
                if not alley or segment.startswith(alley + "/"):
                    alley = segment
                elif segment != alley and not alley.endswith("/" + segment):
                    alley += "/" + segment
            alley = re.sub(r"\b0+(?=\d)", "", alley)
            house_number = re.search(r"\b(?:so(?: nha)?|sn)\s*(\d+[a-z]?(?:/\d+[a-z]?)*)(?!\w)", preceding_text)
            if not house_number:
                house_number = re.search(r"(?<!\w)(\d+[a-z]?(?:/\d+[a-z]?)*)\s*$", preceding_text) if not alley else re.search(r"(?<!\w)(\d+[a-z]?)\s+(?:ngo|ngach|hem)\b", preceding_text)
            data.at[index, "_ngo"] = alley
            data.at[index, "_so_nha"] = re.sub(r"\b0+(?=\d)", "", house_number[1]) if house_number else ""
        if len(districts) == 1:
            directly_matched_district[index] = next(iter(districts))

    # Chỉ lấy tin có quận trực tiếp từ nguồn/văn bản làm mốc; không dùng kết quả suy diễn để tự củng cố.
    # geocoded_house: khớp số nhà/ngõ/đường OSM, tọa độ đối chiếu ranh giới GADM 4.1.
    # Ranh giới: https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_2.json
    reference_rows = data.loc[previously_known & data["district_method"].isin(["breadcrumb", "cached_breadcrumb", "live_breadcrumb", "cached_breadcrumb_id", "metadata", "geocoded_house"])]
    lookup_by_house, lookup_by_alley = {}, {}
    for index, listing in reference_rows.iterrows():
        street, alley, number = listing["_duong_doi_soat"], listing["_ngo"], listing["_so_nha"]
        if not street:
            continue
        # Nhãn website mâu thuẫn với bảng địa danh không được dùng làm bằng chứng cho nhà khác.
        if listing["Quận/Huyện"] not in districts_by_street[street]:
            continue
        for kind, value in [("ward", listing["_phuong_doi_soat"]), ("phone", listing["phone"])]:
            if number and value not in ("", "chua_ro"):
                lookup_key = (street, alley, number, kind, value)
                lookup_by_house.setdefault(lookup_key, set()).add(listing["Quận/Huyện"])
        if alley and listing["_phuong_doi_soat"] != "chua_ro":
            lookup_key = (street, alley, listing["_phuong_doi_soat"])
            lookup_by_alley.setdefault(lookup_key, []).append((listing["url"], listing["Quận/Huyện"]))
    previous_missing_count = (~previously_known).sum()
    for index, listing in data.loc[~previously_known].iterrows():
        street, alley, number = listing["_duong_doi_soat"], listing["_ngo"], listing["_so_nha"]
        candidates, method = set(), ""
        if index in directly_matched_district:
            candidates, method = {directly_matched_district[index]}, "address_district"
        if not candidates and street and number:
            for kind, value in [("ward", listing["_phuong_doi_soat"]), ("phone", listing["phone"])]:
                candidates |= lookup_by_house.get((street, alley, number, kind, value), set())
            method = "same_house_street"
        if not candidates and street and alley:
            matching_listings = lookup_by_alley.get((street, alley, listing["_phuong_doi_soat"]), [])
            if len({listing_url for listing_url, district_label in matching_listings}) >= 2:
                candidates = {district_label for listing_url, district_label in matching_listings}
                method = "same_alley_street"
        if not candidates and street:
            candidates = districts_by_street[street]
            method = "official_locality_reference" if street == "me tri ha" else "official_street_reference"
        # Chỉ có một quận phù hợp mới điền. Giữ nguyên bản ghi khi mơ hồ hoặc có bằng chứng trái nhau.
        street_districts = districts_by_street.get(street, set())
        if len(candidates) != 1 or (street_districts and candidates.isdisjoint(street_districts)):
            continue
        data.at[index, "Quận/Huyện"] = next(iter(candidates))
        data.at[index, "district_method"] = method
        if street and listing["Đường"] in ("", "Chua_ro", "chính", "Chính", "nội bộ"):
            data.at[index, "Đường"] = official_street_names[street]
        data.at[index, "recovery_note"] += f"district:{method};road={street};alley={alley};house={number};"
    print(f"Khôi phục thêm quận từ địa chỉ/đường/số nhà: {previous_missing_count - data['Quận/Huyện'].eq('Chua_ro').sum():,} tin.")
    return data


def main():
    raw = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)
    data = normalize_raw(raw)
    data = filter_rental_rooms(data)
    data = clean_price_and_area(data)
    data = normalize_dates_and_phones(data)
    data, district_by_name = reconcile_locations(raw, data)
    data = deduplicate_and_cross_check(data)
    data = recover_districts_from_address(data, district_by_name)

    # Nhóm để sau này tránh cùng tài sản xuất hiện cả train và test; không dùng giá.
    group_key = data["source"] + "|" + data["url"]
    has_specific_address = data["phone"].ne("") & data["_address"].str.contains(r"\d") & data["_address"].str.len().ge(15)
    group_key.loc[has_specific_address] = data.loc[has_specific_address, "phone"] + "|" + data.loc[has_specific_address, "_address"] + "|" + data.loc[has_specific_address, "area"].fillna(-1).astype(str)
    data["unit_group"] = "g_" + pd.util.hash_pandas_object(group_key, index=False).astype(str)
    data["PHAN_LOAI"] = "phong_tro"

    # Giữ thứ tự cột của CSV clean để các bước sau đọc đúng schema.
    kept_columns = ["source", "post_id", "url", "title", "description", "address", "phone", "posted_date", "price", "area", "Đường", "Phường/Xã", "Quận/Huyện", "district_method", "PHAN_LOAI", "area_was_missing", "unit_group", "recovery_note"]
    result = data[kept_columns].sort_values("posted_date").reset_index(drop=True)
    result.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig", date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    print(f"\nĐã lưu {len(result):,} tin, {len(kept_columns)} cột: {CLEAN_FILE}")
    print(result["district_method"].value_counts().to_string())
    print(f"Quận chưa rõ: {result['Quận/Huyện'].eq('Chua_ro').sum():,}; diện tích còn thiếu: {result['area'].isna().sum():,}.")
    print("Diện tích thiếu để bước train điền trung vị từ train. Các tiện ích sẽ trích ở feature_eng.py.")


if __name__ == "__main__":
    main()
