# Trọ Giá · Pipeline dự đoán giá phòng trọ Hà Nội

Trọ Giá thu thập tin phòng trọ, chuẩn bị dữ liệu huấn luyện và dùng CatBoost để ước tính giá thuê theo tháng. Web Streamlit hiển thị giải thích SHAP cho từng dự đoán. Chatbot Gemini hỗ trợ hỏi đáp về thuê phòng và giải thích kết quả người dùng chọn gửi kèm.

## 1. Sơ đồ pipeline tổng quan

~~~text
DỮ LIỆU VÀ HUẤN LUYỆN

(Phongtro123 / Phongtro.vn / Thuephongtro)
  -> (Ba script crawl) -> (data/data_raw.csv)
  -> (data_clean.py) -> (data/data_clean.csv)
  -> (feature_eng.py) -> (data/data_ready.csv)
  -> (train_model.py) -> (models/best_model.pkl)

(data_raw.csv + data_clean.csv + data_ready.csv + best_model.pkl)
  -> (final_report_eda.ipynb) -> (EDA trước/sau, tiện ích và SHAP)
(data_ready.csv + best_model.pkl + Web App)
  -> (evaluate_model.py) -> (Chỉ số test, phân tích lỗi và kiểm thử trình duyệt)

TRIỂN KHAI VÀ TƯƠNG TÁC

(run.py) -> (Máy chủ Streamlit) -> (app/web_app.py)
(data_ready.csv) -> (Danh sách địa danh cho form)

(Người dùng) -> (Form Streamlit) -> (Kiểm tra và tạo đặc trưng)
  -> (Pipeline trong best_model.pkl) -> (Giá thuê + SHAP) -> (Kết quả trên web)

(Người dùng) -> (Tab Chatbot Gemini) -> (Máy chủ Streamlit)
  -> (Gemini API) -> (Câu trả lời trên web)
(Giá thuê + SHAP) -> (Ngữ cảnh chatbot, khi người dùng bật gửi kèm)
~~~

## 2. Chi tiết từng giai đoạn

### 2.1. Thu thập dữ liệu

- **Tên giai đoạn** · Data Collection.
- **File xử lý** · [crawl_phongtro123.py](src/crawl_phongtro123.py), [crawl_phongtrovn.py](src/crawl_phongtrovn.py), [crawl_thuephongtro.py](src/crawl_thuephongtro.py).
- **Input** · Trang danh sách và trang chi tiết tin cho thuê tại Hà Nội trên Phongtro123, Phongtro.vn và Thuephongtro.
- **Vai trò cốt lõi** · Web scraping theo từng nguồn và gom các trường như tiêu đề, mô tả, giá, diện tích, địa chỉ, ngày đăng và mã tin về cùng cấu trúc.
- **Output** · [data_raw.csv](data/data_raw.csv) chứa dữ liệu thô từ ba nguồn.

### 2.2. Làm sạch dữ liệu

- **Tên giai đoạn** · Data Cleaning.
- **File xử lý** · [data_clean.py](src/data_clean.py).
- **Input** · `data/data_raw.csv`.
- **Vai trò cốt lõi** · Lọc phạm vi phòng trọ, loại tin trùng và các giá thuê không hợp lệ, chuẩn hóa giá về VNĐ/tháng, diện tích về m² và ngày đăng về định dạng thời gian. Script đối soát tên đường, phường/xã và quận/huyện từ thông tin trong tin, giữ cờ diện tích thiếu và tạo `unit_group` để nhận diện nhóm tin cùng căn khi chia tập.
- **Output** · [data_clean.csv](data/data_clean.csv).

### 2.3. Phân tích khám phá dữ liệu

- **Tên giai đoạn** · Exploratory Data Analysis (EDA).
- **File xử lý** · [final_report_eda.ipynb](src/final_report_eda.ipynb).
- **Input** · Dữ liệu Raw, Clean, Ready và mô hình đã lưu.
- **Vai trò cốt lõi** · Khảo sát dữ liệu thiếu, phân phối giá và diện tích, đồng thời đối chiếu trước và sau làm sạch qua histogram, boxplot và phân tích tương quan Pearson giữa giá và diện tích. Notebook còn so sánh các nhóm tiện ích NLP và xếp hạng SHAP trên test.
- **Output** · Bảng thống kê và biểu đồ hiển thị trong notebook.

### 2.4. Tạo đặc trưng

- **Tên giai đoạn** · Feature Engineering.
- **File xử lý** · [feature_eng.py](src/feature_eng.py).
- **Input** · `data/data_clean.csv`.
- **Vai trò cốt lõi** · Tách năm và tháng đăng, chuẩn hóa nhãn địa danh và dùng regex để trích xuất tiện ích từ tiêu đề, mô tả. Văn bản đưa vào mô hình được che thông tin giá. Cột `price` là nhãn, còn `posted_date` và `unit_group` phục vụ chia tập.
- **Output** · [data_ready.csv](data/data_ready.csv) chứa đặc trưng, nhãn và các trường chia tập.

### 2.5. Huấn luyện và chọn mô hình

- **Tên giai đoạn** · Model Training & Tuning.
- **File xử lý** · [train_model.py](src/train_model.py).
- **Input** · `data/data_ready.csv`.
- **Vai trò cốt lõi** · Chia train, validation và test theo thời gian, đồng thời tách các nhóm tin cùng căn. Pipeline học phép điền khuyết thiếu bằng trung vị (**median imputation**) từ train. Phần mã hóa biến phân loại (**categorical encoding**) kết hợp xử lý danh mục của CatBoost với **target encoding** có làm trơn theo quận/phường/đường và cross-fitting theo nhóm. Các phương án biểu diễn văn bản và mã hóa vị trí được so sánh bằng RMSE validation. Siêu tham số đã chốt gồm độ sâu 6, 600 cây và tốc độ học 0,05. Phần văn bản dùng BoW với từ đơn và cụm hai từ.
- **Output** · [best_model.pkl](models/best_model.pkl) chứa bộ tiền xử lý đã fit và CatBoostRegressor trong cùng một pipeline.

### 2.6. Đánh giá mô hình và triển khai

- **Tên giai đoạn** · Model & Deployment Evaluation.
- **File xử lý** · [evaluate_model.py](src/evaluate_model.py).
- **Input** · `data/data_ready.csv`, `models/best_model.pkl` và Web App khi bật phép đo trình duyệt.
- **Vai trò cốt lõi** · Tái lập tập test theo thời gian, so sánh với baseline trung vị bằng RMSE, MAE và R², liệt kê các tin có sai số lớn. Chế độ đo web kiểm tra ràng buộc đầu vào, độ trễ từ lúc bấm đến lúc giá xuất hiện và tính nhất quán với dự đoán trực tiếp.
- **Output** · Chỉ số và kết quả kiểm thử trên terminal. Bảng tổng hợp, biểu đồ và phân tích lỗi nằm trong [báo cáo đồ án](Bao_cao_Do_an.pdf).

### 2.7. Triển khai Web App và chatbot

- **Tên giai đoạn** · Deployment & User Interaction.
- **File xử lý** · [run.py](run.py) khởi chạy Streamlit. [web_app.py](app/web_app.py) xử lý giao diện, dự đoán và chat.
- **Input** · `models/best_model.pkl`, danh sách địa danh từ `data/data_ready.csv`, thông tin phòng và câu hỏi do người dùng nhập. Khóa Gemini được đọc từ `app/api_key.py` ở phía máy chủ.
- **Vai trò cốt lõi** · Kiểm tra các trường bắt buộc và lọc địa danh theo lựa chọn, tạo đặc trưng cùng định dạng với lúc train, gọi pipeline đã lưu và tính đóng góp SHAP. Tab chatbot gọi Gemini API khi người dùng gửi câu hỏi.
- **Output** · Giá thuê ước tính theo tháng, giải thích SHAP và câu trả lời chatbot hiển thị trên trình duyệt.

## 3. Luồng tương tác người dùng

1. **Nhập thông tin.** Người dùng chọn diện tích, quận/huyện và ngày đăng hợp lệ. Phường/xã, đường, tiện ích và mô tả có thể bổ sung. Loại phòng cố định là phòng trọ.
2. **Gửi yêu cầu dự đoán.** Khi bấm **Ước tính giá thuê**, trình duyệt gửi trạng thái widget qua kết nối Streamlit đến tiến trình Python. Điểm xử lý dự đoán là hàm `predict_room_price()` trong `app/web_app.py`.
3. **Chạy mô hình và hiển thị kết quả.** `build_room_features()` dựng một hàng đặc trưng theo schema đã huấn luyện. Pipeline trong `best_model.pkl` áp dụng bộ tiền xử lý đã fit rồi gọi CatBoost. Streamlit hiển thị giá thuê và các khoản đóng góp SHAP theo nhóm.
4. **Hỏi chatbot.** Khi người dùng gửi tin ở tab **Chatbot Gemini**, hàm `ask_gemini()` gọi Gemini API từ máy chủ. Yêu cầu gồm câu hỏi và lịch sử gần đây. Nếu bật **Gửi kèm kết quả ước tính**, yêu cầu có thêm giá CatBoost và tóm tắt SHAP. Phản hồi Gemini xuất hiện trong khung chat, còn giá CatBoost hiển thị ở tab dự đoán.

## Cài đặt và chạy

Mở terminal trong thư mục dự án bằng Python 3.12.

~~~powershell
python -m pip install -r requirements.txt
python run.py
~~~

Trình duyệt mở tại [127.0.0.1:8501](http://127.0.0.1:8501). Để dùng chatbot, khai báo biến `GEMINI_API_KEY` trong `app/api_key.py`. Bản nộp có thể dùng giá trị mẫu và lưu khóa thật trên máy cá nhân.

## Tái chạy đánh giá

~~~powershell
python src/evaluate_model.py
python -m pip install playwright==1.64.1
python src/evaluate_model.py --web --runs 100 --browser msedge
~~~

Lượt đo trình duyệt sử dụng Microsoft Edge đã cài trên máy. Thêm cờ `--gemini` để gửi một câu hỏi thử đến Gemini. Khi chuyển máy và chạy lại các bước cào, làm sạch hoặc huấn luyện, cập nhật đường dẫn CSV và mô hình đã khai báo trong các script tương ứng.
