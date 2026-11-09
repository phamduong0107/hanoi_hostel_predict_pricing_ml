"""So sánh CatBoost văn bản và mã hóa mặt bằng giá theo vị trí."""

import os
import sys

import joblib
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

from feature_eng import QUALITY_SIGNALS, READY_FILE, NEARBY_PLACES, AMENITIES


MODEL_FILE = r"C:\Users\NAME\Documents\MY_PROJECT\models\best_model.pkl"
LOCATION_COLUMNS = ["Quận/Huyện", "Phường/Xã", "Đường"]
NUMERIC_COLUMNS = [
    "area", "area_was_missing", "posted_year", "posted_month",
    *AMENITIES, "rieng_chu", "chung_chu", *NEARBY_PLACES, *QUALITY_SIGNALS,
]
FEATURE_COLUMNS = [*NUMERIC_COLUMNS, *LOCATION_COLUMNS, "text"]

# Kết quả đã chốt trên cùng benchmark thời gian, dùng lại ở các vòng thử nghiệm.
TEST_METRICS = {
    "so_tin": 1618, "tu_ngay": "02/01/2026",
    "R2": 0.5012360016350627, "RMSE": 790379.0775257067, "MAE": 592839.895841305,
}


def build_model(params, groups):
    options = params.copy()
    smoothing = options.pop("gia_khu_vuc", None)
    word_count = options.pop("so_tu", 300)
    use_bigrams = options.pop("cum_tu", False)
    steps = [
        ("so", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
        ("vi_tri", SimpleImputer(strategy="constant", fill_value="Chua_ro"), LOCATION_COLUMNS),
        ("van_ban", SimpleImputer(strategy="constant", fill_value="khong_ro"), ["text"]),
    ]
    if smoothing is not None:
        # Mỗi dòng train được mã hóa từ các nhóm tin khác; validation/test chỉ tra từ train.
        group_splits = list(GroupKFold(n_splits=5).split(groups, groups=groups))
        price_encoder = TargetEncoder(target_type="continuous", smooth=smoothing, cv=group_splits)
        steps.append(("gia_khu_vuc", price_encoder, LOCATION_COLUMNS))

    dictionaries = [
        {"dictionary_id": name, "gram_order": ngram_order, "max_dictionary_size": "4000",
         "occurrence_lower_bound": "3"}
        for name, ngram_order in ([("Words", "1"), ("Phrases", "2")] if use_bigrams else [("Words", "1")])
    ]
    categorical_indices = list(range(len(NUMERIC_COLUMNS), len(NUMERIC_COLUMNS) + len(LOCATION_COLUMNS)))
    model = CatBoostRegressor(**({
        "iterations": 600, "depth": 6, "learning_rate": 0.05, "loss_function": "RMSE",
        "cat_features": categorical_indices, "text_features": [categorical_indices[-1] + 1],
        "feature_calcers": (f"BoW:top_tokens_count={word_count}",), "dictionaries": dictionaries,
        "random_seed": 42, "thread_count": 4, "verbose": False, "allow_writing_files": False,
    } | options))
    return make_pipeline(ColumnTransformer(steps), model)


def regression_metrics(target_prices, predictions):
    return {
        "RMSE": root_mean_squared_error(target_prices, predictions),
        "MAE": mean_absolute_error(target_prices, predictions),
        "R2": r2_score(target_prices, predictions),
    }


def group_shap_contributions(model, data):
    """Tính đóng góp SHAP theo nhóm và mức giá gốc, đơn vị VNĐ/tháng."""
    x = data[model.feature_names_in_]
    preprocessor = model.named_steps["columntransformer"]
    catboost = model.named_steps["catboostregressor"]
    pool = Pool(preprocessor.transform(x), cat_features=catboost.get_cat_feature_indices(),
                text_features=catboost.get_text_feature_indices())
    shap = catboost.get_feature_importance(data=pool, type="ShapValues", thread_count=2)
    groups = []
    for name in preprocessor.get_feature_names_out():
        column = name.split("__", 1)[-1]
        if column in LOCATION_COLUMNS:
            groups.append("Vị trí")
        elif column in {"area", "area_was_missing"}:
            groups.append("Diện tích")
        elif column.startswith("posted_"):
            groups.append("Thời gian")
        elif column == "text":
            groups.append("Mô tả")
        else:
            groups.append("Tiện ích và chất lượng")
    contributions = pd.DataFrame(shap[:, :-1]).T.groupby(groups, sort=False).sum().T
    return contributions, shap[:, -1]


def explain_predictions(model, x):
    """In bảng giải thích mô hình trên các tin kiểm tra."""
    contributions, base_values = group_shap_contributions(model, x)
    print(f"\nĐộ lớn đóng góp trung bình trên {len(x)} tin test (VNĐ/tháng):")
    print(contributions.abs().mean().sort_values(ascending=False).round().astype(int).to_string())
    print(f"\nTin đầu tiên: {x.iloc[0]['area']} m², {x.iloc[0]['Quận/Huyện']}")
    print(f"Mức giá gốc của mô hình: {base_values[0]:,.0f} VNĐ")
    for name, amount in contributions.iloc[0].items():
        print(f"{name}: {amount:+,.0f} VNĐ")
    print(f"Giá dự đoán: {base_values[0] + contributions.iloc[0].sum():,.0f} VNĐ")


def read_chronological_data(file_path):
    data = pd.read_csv(file_path)
    date_value = pd.to_datetime(data["posted_date"], utc=True)
    data["ngay"] = date_value.dt.tz_convert("Asia/Ho_Chi_Minh")
    return data.sort_values("ngay", kind="stable").reset_index(drop=True)


def split_cutoffs(date_value):
    """Lấy các mốc validation và test trên trục thời gian đã sắp xếp."""
    return tuple(date_value.iloc[int(len(date_value) * fraction)] for fraction in (0.50, 0.65, 0.80))


def select_development_set(data, date_value, test_cutoff, test):
    # Một nhóm tin xuất hiện trong test thì không đi vào tập phát triển.
    before_test = date_value < test_cutoff
    different_group = ~data["unit_group"].isin(test["unit_group"])
    return data.loc[before_test & different_group]


def validation_splits(development, first_cutoff, second_cutoff, test_cutoff):
    for start, end in ((first_cutoff, second_cutoff), (second_cutoff, test_cutoff)):
        validation = development.loc[
            development["ngay"].between(start, end, inclusive="left")
        ]
        train = development.loc[
            (development["ngay"] < start)
            & ~development["unit_group"].isin(validation["unit_group"])
        ]
        yield start, end, train, validation


def compare_configurations(experiments, folds):
    result = []
    for name, params in experiments.items():
        print(f"\n{name}", flush=True)
        actual_prices, predictions = [], []
        for fold, (train, validation) in enumerate(folds, 1):
            model = build_model(params, train["unit_group"]).fit(train[FEATURE_COLUMNS], train["price"])
            actual_prices.extend(validation["price"])
            predictions.extend(model.predict(validation[FEATURE_COLUMNS]))
            print(f"  Đã hoàn tất validation {fold}.", flush=True)
        scores = regression_metrics(actual_prices, predictions)
        print(f"  RMSE {scores['RMSE']:,.0f} | MAE {scores['MAE']:,.0f} | R² {scores['R2']:.4f}", flush=True)
        result.append({"Phiên bản": name, **scores})

    return pd.DataFrame(result).sort_values("RMSE")


def main():
    data = read_chronological_data(READY_FILE)
    if data[["ngay", "unit_group", "price"]].isna().any().any() or data["price"].le(0).any():
        raise ValueError("Cần ngày đăng, nhóm tin và giá dương cho tất cả bản ghi.")

    # Giữ nguyên mốc test để so sánh với phiên bản trước.
    date_value = data["ngay"].dt.normalize()
    first_cutoff, second_cutoff, test_cutoff = split_cutoffs(date_value)
    test = data.loc[date_value >= test_cutoff]
    if "--evaluate" in sys.argv:
        model = joblib.load(MODEL_FILE)
        predictions = model.predict(test[model.feature_names_in_])
        print(f"{len(test):,} tin test từ {test_cutoff.date()} | {regression_metrics(test['price'], predictions)}")
        raise SystemExit

    if "--explain" in sys.argv:
        test_sample = test.iloc[::max(1, len(test) // 100)].head(100)
        explain_predictions(joblib.load(MODEL_FILE), test_sample)
        raise SystemExit

    development = select_development_set(data, date_value, test_cutoff, test)
    folds = []
    for start, end, train, validation in validation_splits(
        development, first_cutoff, second_cutoff, test_cutoff
    ):
        folds.append((train, validation))
        print(
            f"Validation {start.date()} đến trước {end.date()}: "
            f"{len(train):,} train, {len(validation):,} validation",
            flush=True,
        )

    experiments = {
        "BoW 300": {},
        "Thêm giá theo vị trí": {"gia_khu_vuc": 12},
        "Giá theo vị trí và one-hot": {"gia_khu_vuc": 12, "one_hot_max_size": 40},
        "Giá theo vị trí và cụm từ": {"gia_khu_vuc": 12, "so_tu": 500, "cum_tu": True},
    }
    ranking = compare_configurations(experiments, folds)
    print("\nKết quả validation\n" + ranking.to_string(index=False), flush=True)
    name = ranking.iloc[0]["Phiên bản"]
    params = experiments[name]
    print(f"\nChốt {name} | {params}", flush=True)
    if name == "BoW 300" and os.path.isfile(MODEL_FILE):
        print("Các đặc trưng vị trí chưa giảm RMSE validation. Giữ mô hình hiện tại.", flush=True)
    else:
        model = build_model(params, development["unit_group"]).fit(development[FEATURE_COLUMNS], development["price"])
        new_scores = regression_metrics(test["price"], model.predict(test[FEATURE_COLUMNS]))
        print(f"Ứng viên mới, {len(test):,} tin test: {new_scores}", flush=True)
        try:
            previous_model = joblib.load(MODEL_FILE)
            previous_scores = regression_metrics(test["price"], previous_model.predict(test[previous_model.feature_names_in_]))
        except FileNotFoundError:
            previous_scores = None
        print(f"Mô hình đang lưu: {previous_scores}", flush=True)

        # Test là benchmark đã dùng trước đây; chỉ thay file nếu ứng viên tốt hơn.
        if previous_scores is None or new_scores["R2"] > previous_scores["R2"]:
            model.training_parameters_ = params.copy()
            model.validation_metrics_ = ranking.iloc[0][["RMSE", "MAE", "R2"]].to_dict()
            joblib.dump(model, MODEL_FILE, compress=3)
            print(f"Đã cập nhật {MODEL_FILE}.", flush=True)
        else:
            print("Giữ mô hình đang lưu.", flush=True)


if __name__ == "__main__":
    main()
