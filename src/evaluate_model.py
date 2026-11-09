"""Đánh giá CatBoost trên tập test theo thời gian và kiểm thử web Streamlit."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from train_model import (
    LOCATION_COLUMNS, regression_metrics, validation_splits, read_chronological_data, split_cutoffs, select_development_set,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "best_model.pkl")
READY = os.path.join(ROOT, "data", "data_ready.csv")
WEB = os.path.join(ROOT, "app", "web_app.py")


def evaluate_model(model):
    data = read_chronological_data(READY)
    normalized_dates = data["ngay"].dt.normalize()
    first_cutoff, second_cutoff, test_cutoff = split_cutoffs(normalized_dates)
    test = data.loc[normalized_dates >= test_cutoff].copy()
    train = select_development_set(data, normalized_dates, test_cutoff, test)
    assert train["ngay"].max() < test["ngay"].min()
    assert set(train["unit_group"]).isdisjoint(test["unit_group"])
    folds = [
        {
            "from": str(start_date.date()), "until": str(end_date.date()),
            "train": len(fold_train), "validation": len(validation),
        }
        for start_date, end_date, fold_train, validation in validation_splits(
            train, first_cutoff, second_cutoff, test_cutoff
        )
    ]

    baseline_train = train.loc[train["posted_year"] >= 2020]
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(baseline_train), 1)), baseline_train["price"])
    predictions = model.predict(test[model.feature_names_in_])
    test["du_doan"] = predictions
    test["sai_so"] = test["du_doan"] - test["price"]
    test["sai_so_tuyet_doi"] = test["sai_so"].abs()
    price_band = pd.cut(test["price"], [0, 2e6, 4e6, 6e6, np.inf], labels=["≤2", "2–4", "4–6", ">6"])
    price_groups = test.groupby(price_band, observed=True).agg(
        so_tin=("price", "size"), mae=("sai_so_tuyet_doi", "mean"), lech=("sai_so", "mean"),
    )
    location_groups = test.groupby(test["Quận/Huyện"].eq("Chua_ro")).agg(
        so_tin=("price", "size"), mae=("sai_so_tuyet_doi", "mean"),
    )
    column = ["area", *LOCATION_COLUMNS, "price", "du_doan", "sai_so_tuyet_doi"]
    return {
        "from": str(test_cutoff.date()), "until": str(test["ngay"].max().date()),
        "train": len(train), "test": len(test), "validation": folds,
        "baseline_train": len(baseline_train), "baseline_price": float(baseline.constant_.ravel()[0]),
        "baseline": regression_metrics(test["price"], baseline.predict(np.zeros((len(test), 1)))),
        "catboost": regression_metrics(test["price"], predictions),
        "price_groups": price_groups.reset_index(names="gia").to_dict("records"),
        "location_groups": location_groups.reset_index(names="thieu_quan").to_dict("records"),
        "worst": json.loads(test.nlargest(20, "sai_so_tuyet_doi")[column].to_json(orient="records")),
    }, test


def check_input_constraints():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(WEB, default_timeout=30).run()
    assert not app.exception
    assert app.button[-2].label == "Ước tính giá thuê" and app.button[-2].disabled
    app.button[0].click().run()
    app.number_input(key="area").set_value(None).run()
    assert app.button[-2].disabled
    app.button[0].click().run()
    app.selectbox(key="district").set_value(None).run()
    assert app.button[-2].disabled
    app.button[0].click().run()
    app.selectbox(key="ward").select("Dịch Vọng Hậu").run()
    assert "Cốm Vòng" in app.selectbox(key="street").options
    app.selectbox(key="street").select("Cốm Vòng").run()
    app.selectbox(key="district").select("Quận Hà Đông").run()
    assert app.selectbox(key="ward").value is None and app.selectbox(key="street").value is None
    app.selectbox(key="posted_year").select(2024).run()
    assert app.selectbox(key="posted_month").value is None and app.button[-2].disabled
    app.selectbox(key="posted_month").select(2).run()
    assert app.selectbox(key="posted_day").options[-1] == "29"
    app.selectbox(key="posted_year").select(2025).run()
    app.selectbox(key="posted_month").select(2).run()
    assert app.selectbox(key="posted_day").options[-1] == "28"
    assert not app.exception


def wait_for_streamlit(server, url):
    for _ in range(120):
        try:
            with urlopen(url + "/_stcore/health", timeout=0.5) as r:
                if r.status == 200:
                    break
        except (URLError, TimeoutError):
            if server.poll() is not None:
                raise RuntimeError("Streamlit thoát khi khởi động.")
            time.sleep(.25)
    else:
        raise RuntimeError("Không mở được máy chủ Streamlit để kiểm thử.")


def check_chatbot(page, expect, test_gemini):
    gemini = "Không chạy trong lượt đo này"
    if test_gemini:
        page.get_by_role("tab", name="Chatbot Gemini").click()
        page.get_by_role("textbox", name="Câu hỏi của bạn").fill(
            "Khi xem phòng trọ, tôi nên hỏi gì về giá điện? Trả lời trong một câu."
        )
        page.get_by_role("button", name="Gửi câu hỏi", exact=True).click()
        try:
            expect(page.get_by_test_id("stChatMessage")).to_have_count(3, timeout=35000)
            expect(page.get_by_test_id("stChatMessage").last).to_contain_text(
                "điện", ignore_case=True, timeout=35000,
            )
            assert page.get_by_test_id("stAlert").count() == 0
            gemini = "Đạt, nhận phản hồi thật từ Gemini"
        except AssertionError:
            gemini = "Chưa xác nhận được phản hồi Gemini trong lượt thử"
    return gemini


def benchmark_web(model, run_count, browser_channel, test_gemini=False):
    from playwright.sync_api import expect, sync_playwright

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from app.web_app import ROOM_EXAMPLES, build_room_features

    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-B", "-m", "streamlit", "run", WEB,
         "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
         "--browser.gatherUsageStats=false"], cwd=ROOT, stdout=subprocess.DEVNULL,
    )
    try:
        wait_for_streamlit(server, url)

        today = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
        rng = np.random.default_rng(42)
        # Diện tích khác nhau giúp kiểm tra mỗi lượt đã dùng đúng giá trị mới.
        sampled_areas = rng.choice(np.arange(120, 601) / 10, size=run_count, replace=False)
        sampled_templates = rng.integers(len(ROOM_EXAMPLES), size=run_count)
        # Chạy theo từng mẫu phòng để tránh cập nhật chồng nhau giữa các widget.
        cases = sorted(zip(sampled_templates.tolist(), sampled_areas.tolist()), key=lambda case: case[0])
        ms_web, ms_model, differences = [], [], []
        examples = {}
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel=browser_channel, headless=True)
            page = browser.new_page()
            page.goto(url)
            predict_button = page.get_by_role("button", name="Ước tính giá thuê", exact=True)
            expect(predict_button).to_be_disabled(timeout=30000)
            area_input = page.get_by_test_id("stNumberInput").locator("input")
            result = page.locator(".tg-price")
            price_locator = page.locator(".tg-price[data-price]")
            previous_group = None
            for i, (template_index, area) in enumerate([(0, ROOM_EXAMPLES[0][2]), *cases]):
                example_label, district, example_area, amenities = ROOM_EXAMPLES[template_index]
                if template_index != previous_group:
                    page.get_by_role("button", name=example_label, exact=True).click()
                    expect(area_input).to_have_value(f"{example_area:.2f}")
                    expect(result).to_have_attribute("data-area", str(example_area))
                    expect(result).to_have_attribute("data-district", district)
                    previous_group = template_index
                if i:
                    area_input.fill(f"{area:.2f}")
                    area_input.press("Tab")
                expect(result).to_have_attribute("data-area", str(area))
                expect(result).to_have_attribute("data-district", district)
                expect(price_locator).to_have_count(0)
                expect(predict_button).to_be_enabled()

                room = dict(
                    area=area, district=district, ward=None, street=None, amenities=amenities, owner="Chưa rõ",
                    title=f"Phòng trọ {district.removeprefix('Quận ')}", description="", date=today,
                )
                x = build_room_features(model, room)
                start = time.perf_counter()
                model_price = float(model.predict(x)[0])
                model_ms = 1000 * (time.perf_counter() - start)
                start = time.perf_counter()
                predict_button.click()
                current_price_locator = price_locator.and_(page.locator(f'[data-area="{area}"][data-district="{district}"]'))
                web_price = float(current_price_locator.get_attribute("data-price", timeout=15000))
                web_ms = 1000 * (time.perf_counter() - start)
                difference = abs(web_price - model_price)
                assert difference <= 1e-6, f"Lượt {i}, diện tích {area}: web {web_price}, trực tiếp {model_price}"
                assert current_price_locator.inner_text() == f"{model_price:,.0f}".replace(",", ".") + " ₫"
                if i == 0:
                    continue
                ms_web.append(web_ms)
                ms_model.append(model_ms)
                differences.append(difference)
                if template_index not in examples:
                    examples[template_index] = {
                        "template": template_index, "district": district, "area": area,
                        "amenities": amenities, "price": round(web_price),
                    }
                if i % 20 == 0:
                    print(f"Đã đối chiếu {i}/{run_count} lượt.", file=sys.stderr, flush=True)

            gemini = check_chatbot(page, expect, test_gemini)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=15)

    check_input_constraints()
    return {
        "n": run_count, "browser": browser_channel, "date": str(today), "seed": 42,
        "mean_ms": float(np.mean(ms_web)), "median_ms": float(np.median(ms_web)),
        "p95_ms": float(np.percentile(ms_web, 95)), "offline_mean_ms": float(np.mean(ms_model)),
        "max_diff": max(differences), "examples": list(examples.values()), "gemini": gemini,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true", help="Đo trực tiếp trên trình duyệt.")
    parser.add_argument("--runs", type=int, choices=range(1, 482), default=100, metavar="1-481")
    parser.add_argument("--browser", default="msedge", help="Kênh Chromium, ví dụ msedge hoặc chrome.")
    parser.add_argument("--gemini", action="store_true", help="Gửi một câu thử tới Gemini khi dùng --web.")
    parser.add_argument("--json", action="store_true", help="In số liệu có cấu trúc ra stdout.")
    args = parser.parse_args()
    model = joblib.load(MODEL)
    report, test = evaluate_model(model)
    if args.web:
        report["web"] = benchmark_web(model, args.runs, args.browser, args.gemini)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    else:
        print(pd.DataFrame([report["baseline"], report["catboost"]], index=["Median", "CatBoost"]))
        print(f"\n{len(test):,} tin test từ {report['from']} đến {report['until']}")
        print("\n20 tin có sai số tuyệt đối cao nhất")
        print(pd.DataFrame(report["worst"]).to_string(index=False))
        if args.web:
            metrics = report["web"]
            print(f"\nWeb {metrics['n']}/{metrics['n']} khớp | Trung bình {metrics['mean_ms']:.1f} ms | P95 {metrics['p95_ms']:.1f} ms")


if __name__ == "__main__":
    main()
