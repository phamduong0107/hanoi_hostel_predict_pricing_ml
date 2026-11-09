"""Cào tin Hà Nội từ Thuephongtro.com; bổ sung vào một file data_raw.csv chung."""

import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Sửa trực tiếp đường dẫn này nếu bạn chuyển dự án sang thư mục khác.
RAW_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\data\data_raw.csv"
PAGE_LIMIT = 0  # 1 = thử một trang; 0 = cào đến hết phân trang.


def read_text(page_number, selector):
    element = page_number.select_one(selector)
    return element.get_text(" ", strip=True) if element is not None else ""


def fetch_html(session, url):
    time.sleep(1)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def parse_listing_page(html):
    page_listings = []
    for listing_element in html.select(".post-listing article.post-item"):
        link_element = listing_element.select_one("a[href]")
        if link_element is None:
            continue
        link = urljoin("https://thuephongtro.com", link_element["href"])
        listing_id = re.search(r"-(\d+)\.html", link)
        listing = {
            "source": "thuephongtro.com",
            "post_id": listing_id.group(1) if listing_id else "",
            "url": link,
            "title": read_text(listing_element, "h3.title"),
            "address": read_text(listing_element, ".post-address"),
            "description": read_text(listing_element, ".post-excerpt"),
            "price": read_text(listing_element, ".price"),
            "area": read_text(listing_element, ".info-features .feature-item"),
            "phone": "",
            "posted_date": read_text(listing_element, ".time"),
            "district_raw": "",
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "detail_status": "pending",
        }
        if listing["title"]:
            page_listings.append(listing)
    return page_listings


def enrich_listing(detail_page, listing):
    description = detail_page.select_one("#overview > .section-content")
    if description is None:
        raise ValueError("Không tìm thấy phần mô tả")
    for element in description.select(".the-post, h2, script, style, iframe"):
        element.decompose()
    for paragraph in description.find_all("p"):
        text_content = paragraph.get_text(" ", strip=True).casefold()
        if text_content.startswith("khu vực:"):
            paragraph.decompose()
    listing["description"] = description.get_text("\n", strip=True)
    # Giá trị ở trang chi tiết được ưu tiên khi có nội dung.
    for column, selector in (
        ("price", "#overview .post-price"),
        ("area", "#overview .post-acreage"),
        ("address", "#overview .post-address span span"),
        ("posted_date", "#overview .post-published"),
    ):
        listing[column] = read_text(detail_page, selector) or listing[column]
    for row in detail_page.select(".section-detail .section-detail-row"):
        if "Ngày cập nhật" in read_text(row, "strong"):
            listing["posted_date"] = read_text(row, "span") or listing["posted_date"]
    district_links = detail_page.select("#breadcrumb a[href]")
    if len(district_links) >= 4:
        listing["district_raw"] = district_links[3].get_text(" ", strip=True)
    phone_link = detail_page.select_one(
        '.detailp-user [data-phone], .detailp-user a[href^="tel:"]'
    )
    if phone_link is not None:
        listing["phone"] = phone_link.get("data-phone", "")
        if not listing["phone"]:
            listing["phone"] = phone_link.get("href", "").removeprefix("tel:")
    listing["detail_status"] = "ok"


if __name__ == "__main__":
    data = []
    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = (
                "Mozilla/5.0 (compatible; RentalResearch/1.0)"
            )
            stop_crawling = False
            page_number = 1
            seen_pages = set()

            while (PAGE_LIMIT == 0 or page_number <= PAGE_LIMIT) and not stop_crawling:
                url = (
                    "https://thuephongtro.com/cho-thue-phong-tro-ha-noi"
                    f"?page={page_number}"
                )
                try:
                    html = fetch_html(session, url)
                except requests.RequestException as error:
                    print(f"Thuephongtro: dừng ở trang {page_number}: {error}", flush=True)
                    break

                page_listings = parse_listing_page(html)
                # So sánh cả trang theo đúng thứ tự URL.
                page_urls = tuple(listing["url"] for listing in page_listings)
                if not page_urls or page_urls in seen_pages:
                    break
                seen_pages.add(page_urls)
                data.extend(page_listings)
                print(
                    f"Thuephongtro | trang {page_number} | {len(page_listings)} tin",
                    flush=True,
                )

                for listing in page_listings:
                    try:
                        enrich_listing(fetch_html(session, listing["url"]), listing)
                    except (requests.RequestException, ValueError) as error:
                        listing["detail_status"] = f"error:{type(error).__name__}"
                        print(
                            f"  Chưa lấy đủ tin {listing['post_id']}: {error}", flush=True
                        )
                        if (
                            isinstance(error, requests.HTTPError)
                            and error.response.status_code in (403, 429)
                        ):
                            stop_crawling = True
                            break
                page_number += 1
    except KeyboardInterrupt:
        print("\nĐã dừng. Lưu những tin đã thu được.", flush=True)
    finally:
        if data:
            # Nối với CSV hiện có để giữ tin từ các nguồn khác.
            try:
                existing_data = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)
            except FileNotFoundError:
                existing_data = pd.DataFrame()

            new_data = pd.DataFrame(data)
            result = pd.concat([existing_data, new_data], ignore_index=True)
            result.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
            print(
                f"Đã thêm {len(new_data)} tin; tổng {len(result)} tin "
                f"trong {RAW_FILE}",
                flush=True,
            )
        else:
            print("Không có tin mới; giữ nguyên dữ liệu đã có.", flush=True)
