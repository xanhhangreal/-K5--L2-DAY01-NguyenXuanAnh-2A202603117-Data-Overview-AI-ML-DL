# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? Cả bức ảnh, vì task là image_classification, không có bounding box.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Từ taxonomy/dataset ImageNet-1K đi kèm mô hình, không phải model tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? Để xác định chính xác mã lớp, tên lớp và hệ thống phân loại tương ứng.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Quy định cách chọn nhãn đại diện khi ảnh có nhiều vật thể hoặc chủ đề.
6. Vì sao không được lấy `score` của model làm ground truth? Vì score chỉ là mức độ dự đoán của model, có thể sai; ground truth cần được con người xác minh theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc trên-trái của box nằm tại (385.33, 69.24), góc dưới-phải tại (498.92, 348.92), box rộng khoảng 113.58 px và cao 279.68 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Ngưỡng 0.35 có 11 predictions, ngưỡng 0.60 có 6 predictions; số lượng ở ngưỡng 0.20 không xác định được từ JSON hiện có.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Khi hạ ngưỡng, reviewer phải kiểm tra thêm nhiều hộp, loại bỏ hộp nhận diện sai, hộp trùng lặp và sửa nhãn hoặc vị trí không chính xác.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). Quy tắc box: Vẽ hộp sát nhất có thể quanh toàn bộ phần vật thể nhìn thấy, hạn chế tối đa phần nền thừa.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? Có một prediction oven nằm sát mép trái ảnh (x1=0.11), có khả năng bị cắt mép; nên hỏi Lab Coach để xác nhận cách xử lý vật thể bị cắt trước khi đưa thành quy tắc chung trong guideline, vì JSON không đủ để xác minh hình ảnh thực tế.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[(446,70), (445,71), (444,71), (443,72), (442,72)].
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? So với box hình chữ nhật, polygon mô tả đường biên thực tế của vật thể chi tiết hơn, giúp phân biệt phần vật thể với nền xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? instance_id là mã định danh riêng cho từng đối tượng trong ảnh, ví dụ kitchen-001 là một cá thể người cụ thể; nó không phải class_id (mã loại đối tượng như person) và cũng không phải tracking_id (mã theo dõi cùng đối tượng qua nhiều khung hình/video).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. Quy tắc vẽ mask: Vẽ sát đường viền phần vật thể thực sự nhìn thấy, không bao gồm nền và không tự suy đoán phần bị che khuất.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? Vị trí ranh giới chưa rõ: Ở khu vực các chiếc bát màu hồng phía góc dưới bên trái, bát nằm trên mặt bàn màu xanh nên ranh giới giữa bowl và dining table khá sát nhau, một phần có thể bị che khuất; xử lý theo guideline bằng cách chỉ vẽ mask theo phần bát nhìn thấy được, tách khỏi mask mặt bàn và không tự suy đoán phần bị che, chỉ escalation nếu ảnh gốc không đủ rõ để xác định ranh giới.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  Một hoặc nhiều nhãn cho toàn bộ ảnh, ví dụ kitchen hoặc các nhãn chủ đề theo taxonomy | Ảnh bếp có nhiều loại vật thể cùng xuất hiện nên cần xác định phân loại một nhãn hay nhiều nhãn; hiện chưa có JSON/PNG prediction riêng cho tác vụ này để xác nhận lỗi cụ thể | Gán nhãn cho toàn ảnh theo guideline và taxonomy; không tự thêm nhãn ngoài quy định | Nhãn có đúng nội dung ảnh không, có thiếu/thừa nhãn không và có nhất quán với taxonomy không. |
| Phát hiện vật thể | Mỗi vật thể là một nhãn gồm class + bounding box (hộp chữ nhật, tọa độ pixel). | JSON kitchen có person score 0.912625, nhưng cũng có person score 0.371262; PNG có nhiều hộp bowl gần nhau, dễ nhầm vật thể hoặc tạo detection trùng. | Vẽ một bounding box cho từng vật thể nhìn thấy, chọn đúng class và xử lý vật thể bị che theo guideline.	| Class có đúng không, hộp có ôm sát vật thể không, có bỏ sót, phát hiện trùng hoặc nhầm vật thể không.   |
| Instance segmentation | Mỗi cá thể vật thể có class + mask/polygon (vùng pixel thuộc vật thể), thường kèm ID riêng. | JSON kitchen-001 là person score 0.899318, polygon có 348 điểm; PNG cho thấy mask dining table màu xanh sát/chồng lên các vùng bowl màu hồng, cần kiểm tra ranh giới giữa hai vật thể. | Vẽ mask theo đường biên từng cá thể, tách các vật thể khác nhau và chỉ gán phần nhìn thấy được. | Mask có bám đúng biên không, có lấn sang nền/vật thể khác không, có tách đúng từng instance và đúng class không. | 

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) Tôi chỉ sử dụng dữ liệu được cấp cho bài lab, không tự ý sao chép, chia sẻ, tải lên nơi khác hoặc tiết lộ thông tin ngoài phạm vi được phép.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? Khi gặp dữ liệu không đúng phạm vi: Tôi sẽ dừng xử lý, không tự ý mở rộng hoặc chia sẻ dữ liệu và báo cho người hướng dẫn (Lab Instructor) hoặc người phụ trách dự án (Project Lead).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
