"""Cào tin Hà Nội từ Phongtro123; bổ sung vào một file data_raw.csv chung."""

import json
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Sửa trực tiếp đường dẫn này nếu bạn chuyển dự án sang thư mục khác.
RAW_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\data\data_raw.csv"
PAGE_LIMIT = 0  # 1 = thử một trang; 0 = cào đến hết phân trang.


# Chỉ một hàm nhỏ: lấy chữ trong một thẻ HTML, thiếu thẻ thì trả chuỗi rỗng.
def read_text(page_number, selector):
    element = page_number.select_one(selector)
    if element is None:
        return ""
    return element.get_text(" ", strip=True)


if __name__ == "__main__":
    data = []  # Chứa các tin mới trên RAM.
    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = "Mozilla/5.0 (compatible; RentalResearch/1.0)"
            stop_crawling = False
            page_number = 1
            seen_pages = []

            while (PAGE_LIMIT == 0 or page_number <= PAGE_LIMIT) and not stop_crawling:
                # Bước 1: Mở trang danh sách.
                url = f"https://phongtro123.com/tinh-thanh/ha-noi?page={page_number}"
                try:
                    time.sleep(1)
                    response = session.get(url, timeout=30)
                    response.raise_for_status()
                    html = BeautifulSoup(response.content, "html.parser")
                except requests.RequestException as error:
                    print(f"Phongtro123: dừng ở trang {page_number}: {error}", flush=True)
                    stop_crawling = True
                    break

                # Bước 2: Đọc các tin. Website lưu thông tin ngay trong thẻ script.
                page_listings = []
                for listing_element in html.select("ul.post__listing li.d-flex.bg-white.shadow-sm.rounded.p-3.mt-3"):
                    script = listing_element.select_one('script[type="application/ld+json"]')
                    if script is None:
                        continue
                    try:
                        metadata = json.loads(script.get_text())  # Chỉ đọc JSON trên RAM.
                    except json.JSONDecodeError:
                        continue
                    address = metadata.get("address", "")
                    if isinstance(address, dict):
                        address = address.get("streetAddress", "")
                    area = ""
                    for element in listing_element.select("div.mb-2.line-clamp-1 span"):
                        text_content = element.get_text("", strip=True)
                        if "m²" in text_content or "m2" in text_content:
                            area = text_content
                            break
                    link = metadata.get("url", "")
                    listing_id = re.search(r"-pr(\d+)\.html", link)
                    listing = {
                        "source": "phongtro123.com",
                        "post_id": listing_id.group(1) if listing_id else "",
                        "url": link,
                        "title": metadata.get("name", ""),
                        "address": address,
                        "description": metadata.get("description", ""),
                        "price": metadata.get("priceRange", ""),
                        "area": area,
                        "phone": metadata.get("telephone", ""),
                        "posted_date": read_text(listing_element, "time"),
                        "district_raw": "",
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "detail_status": "pending",
                    }
                    if listing["url"] and listing["title"]:
                        page_listings.append(listing)

                # Dừng khi hết tin hoặc website trả lại một trang đã đọc.
                page_urls = [listing["url"] for listing in page_listings]
                if not page_listings or page_urls in seen_pages:
                    break
                seen_pages.append(page_urls)
                data.extend(page_listings)
                print(f"Phongtro123 | trang {page_number} | {len(page_listings)} tin", flush=True)

                # Bước 3: Vào từng tin để lấy mô tả đầy đủ và quận.
                for listing in page_listings:
                    try:
                        time.sleep(1)
                        response = session.get(listing["url"], timeout=30)
                        response.raise_for_status()
                        detail_page = BeautifulSoup(response.content, "html.parser")
                        description_heading = detail_page.find("h2", string=re.compile("Thông tin mô tả", re.I))
                        if description_heading is None:
                            raise ValueError("Không tìm thấy phần mô tả")
                        sections = description_heading.parent.find_all("p")
                        listing["description"] = "\n".join(paragraph.get_text(" ", strip=True) for paragraph in sections)
                        for link in detail_page.select(".breadcrumb a[href]"):
                            if re.search(r"/(?:quan|huyen|thi-xa)-", link["href"]):
                                listing["district_raw"] = link.get_text(" ", strip=True)
                        listing["detail_status"] = "ok"
                    except (requests.RequestException, ValueError) as error:
                        listing["detail_status"] = "error:" + type(error).__name__
                        print(f"  Chưa lấy đủ tin {listing['post_id']}: {error}", flush=True)
                        if isinstance(error, requests.HTTPError) and error.response.status_code in (403, 429):
                            stop_crawling = True
                            break
                page_number += 1
    except KeyboardInterrupt:
        print("\nĐã dừng. Lưu những tin đã thu được.", flush=True)
    finally:
        if data:
            # Đọc dữ liệu đã có để giữ nguyên tin của các nguồn khác.
            try:
                existing_data = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)
            except FileNotFoundError:
                existing_data = pd.DataFrame()

            # Nối trên RAM rồi ghi lại đúng một file CSV, không tạo file tạm.
            new_data = pd.DataFrame(data)
            result = pd.concat([existing_data, new_data], ignore_index=True)
            result.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
            print(f"Đã thêm {len(new_data):,} tin; tổng {len(result):,} tin trong {RAW_FILE}", flush=True)
        else:
            print("Không có tin mới; giữ nguyên dữ liệu đã có.", flush=True)
