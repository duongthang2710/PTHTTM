# Kịch bản thuyết trình Compact ResNet trên CIFAR-100

Thời lượng dự kiến: 7–10 phút. Tài liệu này viết cho người mới, có thể đọc gần như trực tiếp khi thuyết trình.

## Slide 1 — Bài toán cần giải quyết

### Nội dung trên slide

- Phân loại ảnh bằng Compact ResNet
- Dữ liệu CIFAR-100 có 100 loại ảnh
- Ảnh đầu vào có kích thước `32 x 32 x 3`
- Mục tiêu: dự đoán đúng loại của ảnh

### Lời thuyết trình

> Phần của em sử dụng Compact ResNet để phân loại ảnh trong bộ dữ liệu CIFAR-100. Bộ dữ liệu này có 100 loại ảnh, ví dụ như động vật, cây cối và phương tiện. Mỗi ảnh có kích thước 32 nhân 32 pixel và có ba kênh màu đỏ, xanh lá, xanh dương. Khi nhận một ảnh, model sẽ đưa ra dự đoán ảnh đó thuộc lớp nào trong 100 lớp.

### Mở và nói về code

```python
CLASS_NAMES = [...]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = x_train.shape[1:]
```

> `CLASS_NAMES` lưu tên của 100 lớp. `NUM_CLASSES` là số lớp đầu ra và có giá trị bằng 100. `INPUT_SHAPE` lấy kích thước của một ảnh đầu vào, ở đây là 32 nhân 32 nhân 3.

---

## Slide 2 — ResNet là gì?

### Nội dung trên slide

```text
Nhánh chính: x → các lớp convolution → F(x)
Nhánh tắt:   x ────────────────────────┐
                                            ↓
Kết quả:                   ReLU(F(x) + x)
```

- ResNet vẫn là một mạng CNN
- Điểm đặc biệt là có đường tắt `shortcut`
- Shortcut giúp thông tin và gradient đi qua mạng dễ hơn

### Lời thuyết trình

> ResNet là một mạng CNN có thêm đường tắt gọi là shortcut connection. Trong CNN thông thường, dữ liệu phải đi lần lượt qua tất cả các lớp. Trong ResNet, đầu vào còn được truyền theo một đường tắt và cộng với kết quả của các lớp convolution. Nhờ đường tắt này, model giữ lại được thông tin ban đầu và việc huấn luyện mạng sâu trở nên ổn định hơn.

### Mở và nói về code

Mở hàm `residual_block_2d` và chỉ vào:

```python
shortcut = x
```

> Dòng này lưu đầu vào `x` vào biến `shortcut`. Đây chính là đường tắt của ResNet.

Tiếp tục chỉ vào:

```python
x = layers.Add(name=f"{name}_add")([x, shortcut])
return layers.Activation("relu", name=f"{name}_out")(x)
```

> `layers.Add` cộng kết quả của nhánh convolution với đầu vào được giữ trong shortcut. Sau phép cộng, dữ liệu đi qua hàm ReLU. Đây là phần quan trọng nhất để phân biệt ResNet với CNN thông thường.

---

## Slide 3 — Một residual block hoạt động thế nào?

### Nội dung trên slide

1. Convolution tìm đặc trưng của ảnh
2. Batch Normalization giúp quá trình học ổn định
3. ReLU giữ lại các tín hiệu dương
4. Cộng kết quả với shortcut

### Lời thuyết trình

> Trong một residual block, ảnh đi qua hai lớp convolution. Có thể hiểu convolution là bộ lọc giúp model tìm các đặc trưng như cạnh, màu sắc và hình dạng. Batch Normalization giúp dữ liệu giữa các lớp ổn định hơn. ReLU thêm tính phi tuyến để model học được những đặc trưng phức tạp. Cuối block, kết quả được cộng với shortcut.

### Mở và nói về code

```python
layers.Conv2D(filters, 3, ...)
layers.BatchNormalization(...)
layers.Activation("relu", ...)
```

> `Conv2D` với kernel bằng 3 nghĩa là model dùng bộ lọc 3 nhân 3. `filters` là số đặc trưng mà model muốn học. Sau đó là Batch Normalization và ReLU.

Chỉ vào điều kiện:

```python
if stride != 1 or int(shortcut.shape[-1]) != filters:
    shortcut = layers.Conv2D(filters, 1, strides=stride, ...)(shortcut)
```

> Hai nhánh chỉ cộng được khi có cùng kích thước. Nếu số kênh hoặc kích thước ảnh thay đổi, code dùng convolution 1 nhân 1 để điều chỉnh shortcut cho phù hợp.

---

## Slide 4 — Cấu trúc toàn bộ model

### Nội dung trên slide

```text
Ảnh 32 x 32 x 3
        ↓
Tiền xử lý và augmentation
        ↓
Convolution đầu tiên: 32 filters
        ↓
Residual stage: 32 → 64 → 128 filters
        ↓
Global Average Pooling + Dropout
        ↓
100 xác suất đầu ra
```

### Lời thuyết trình

> Đầu tiên, ảnh được chuẩn hóa và biến đổi nhẹ để model có thêm dữ liệu học. Sau đó ảnh đi qua một lớp convolution đầu tiên. Model có ba nhóm residual, lần lượt dùng 32, 64 và 128 filters. Số filters tăng dần vì các lớp sau cần học những đặc trưng phức tạp hơn. Cuối cùng, model trả về 100 xác suất tương ứng với 100 lớp.

### Mở và nói về code

Mở hàm `build_resnet_compact`:

```python
x = conv2d_bn_relu(x, 32, name="stem")

for stage, filters in enumerate([32, 64, 128], start=1):
    stride = 1 if stage == 1 else 2
    x = residual_block_2d(x, filters, stride=stride, name=f"res_s{stage}_b1")
    x = residual_block_2d(x, filters, stride=1, name=f"res_s{stage}_b2")
```

> Vòng lặp tạo ba stage với 32, 64 và 128 filters. Mỗi stage có hai residual block. Từ stage thứ hai, block đầu dùng stride bằng 2 để giảm chiều rộng và chiều cao của feature map, đồng thời tăng số đặc trưng.

Chỉ tiếp vào:

```python
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.35, name="head_dropout")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

> Global Average Pooling thu gọn các đặc trưng thành một vector. Dropout tạm thời bỏ ngẫu nhiên 35 phần trăm neuron khi train để hạn chế học thuộc. Lớp Dense cuối dùng softmax để tạo 100 xác suất có tổng bằng 1.

Lưu ý khi nói: đây là **Compact ResNet tự xây dựng**, không phải ResNet-18 hay ResNet-50.

---

## Slide 5 — Chuẩn bị dữ liệu

### Nội dung trên slide

| Tập dữ liệu | Số ảnh | Mục đích |
|---|---:|---|
| Train | 45.000 | Cho model học |
| Validation | 5.000 | Theo dõi trong lúc train |
| Test | 10.000 | Đánh giá cuối cùng |

### Lời thuyết trình

> Dữ liệu được chia thành ba phần. Tập train dùng để cập nhật model. Tập validation cho biết model có hoạt động tốt trên dữ liệu chưa nhìn thấy trong quá trình học hay không. Tập test chỉ được dùng sau khi train xong để đánh giá cuối cùng. Việc tách riêng như vậy giúp kết quả khách quan hơn.

### Mở và nói về code

```python
train_idx, val_idx = train_test_split(
    full_idx,
    test_size=5000,
    random_state=SEED,
    stratify=y_train_full,
)
```

> `test_size=5000` lấy 5.000 ảnh làm validation. `stratify` giữ số lượng mẫu của các lớp cân bằng khi chia. `random_state` giúp lần chạy sau vẫn chia dữ liệu giống lần trước.

Mở `make_image_preprocessor` và nói:

> Code chia pixel cho 255, chuẩn hóa ảnh và tạo các biến thể bằng lật, xoay, dịch chuyển và zoom. Các biến đổi này chỉ hoạt động khi train, giúp model bớt học thuộc dữ liệu.

---

## Slide 6 — Model được huấn luyện như thế nào?

### Nội dung trên slide

- AdamW: thuật toán cập nhật trọng số
- Loss: đo mức độ dự đoán sai
- Batch size: 64 ảnh mỗi lần cập nhật
- Tối đa: 20 epochs
- Theo dõi `val_loss` để điều chỉnh việc học

### Lời thuyết trình

> Huấn luyện nghĩa là cho model xem ảnh, so sánh dự đoán với nhãn đúng, sau đó điều chỉnh các trọng số để lần sau dự đoán tốt hơn. Một epoch là một lần model học qua toàn bộ tập train. Trong bài này model học tối đa 20 epoch, mỗi lần xử lý 64 ảnh.

### Mở và nói về code

```python
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
```

> `AdamW` là thuật toán điều chỉnh trọng số. `learning_rate` quyết định mỗi lần model thay đổi trọng số nhiều hay ít. `weight_decay` giúp hạn chế model trở nên quá phức tạp. Loss đo mức độ dự đoán sai, còn accuracy đo tỷ lệ dự đoán đúng.

Mở tiếp:

```python
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
)
```

> `model.fit` là lệnh bắt đầu huấn luyện. `x_train` là ảnh và `y_train` là nhãn đúng. Sau mỗi epoch, model được kiểm tra trên tập validation. Kết quả từng epoch được lưu trong `history` để vẽ biểu đồ.

Mở callbacks và nói:

> Nếu validation loss không cải thiện, `ReduceLROnPlateau` giảm learning rate để model học chậm và cẩn thận hơn. `EarlyStopping` có thể dừng quá trình học và khôi phục bộ trọng số tốt nhất.

---

## Slide 7 — Model dự đoán như thế nào?

### Nội dung trên slide

```text
Một ảnh test
    ↓
Model tạo 100 xác suất
    ↓
Chọn lớp có xác suất cao nhất
```

### Lời thuyết trình

> Sau khi train, model được dùng để dự đoán tập test. Với mỗi ảnh, model trả về 100 xác suất. Chương trình chọn vị trí có xác suất lớn nhất làm lớp dự đoán.

### Mở và nói về code

```python
y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = y_prob.argmax(axis=1)
```

> `y_prob` chứa xác suất của 100 lớp cho từng ảnh. `argmax` tìm lớp có xác suất lớn nhất. Ví dụ, nếu xác suất lớn nhất nằm ở vị trí 25 thì model dự đoán ảnh thuộc lớp số 25.

Mở tiếp:

```python
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)
```

> Code không chỉ tính accuracy mà còn tính Precision, Recall và F1 cho từng lớp rồi lấy trung bình. Cách tính Macro giúp cả 100 lớp được xem là quan trọng như nhau.

---

## Slide 8 — Kết quả chạy và nhận xét

### Nội dung trên slide

| Chỉ số | Compact ResNet |
|---|---:|
| Accuracy | 50,62% |
| Macro Precision | 56,65% |
| Macro Recall | 50,62% |
| Macro F1 | 50,55% |
| Test Loss | 1,8785 |
| Số tham số | 710.468 |
| Best epoch | 20 |
| Thời gian train | khoảng 2.018,57 giây |
| Thời gian dự đoán 10.000 ảnh | khoảng 6,70 giây |

### Lời thuyết trình

> Compact ResNet đạt accuracy 50,62 phần trăm. Điều này có nghĩa là trong 10.000 ảnh test, model dự đoán đúng khoảng 5.062 ảnh. Với bài toán có tới 100 lớp, đoán ngẫu nhiên chỉ đạt khoảng 1 phần trăm, vì vậy kết quả 50,62 phần trăm cho thấy model đã học được nhiều đặc trưng có ý nghĩa.
>
> Macro F1 đạt 50,55 phần trăm và khá gần accuracy. Điều này cho thấy kết quả không chỉ đến từ một vài lớp dễ mà tương đối trải đều trên nhiều lớp. Macro Precision cao hơn, đạt 56,65 phần trăm, nghĩa là khi model chọn một lớp thì dự đoán thường khá chính xác; tuy nhiên model vẫn bỏ sót nhiều ảnh của một số lớp nên Macro Recall chỉ đạt 50,62 phần trăm.
>
> Model có khoảng 710 nghìn tham số, tức là có 710 nghìn giá trị được học trong quá trình train. Best epoch bằng 20, cũng là epoch cuối cùng. Điều này cho thấy model vẫn có thể còn cải thiện nếu được train thêm, vì early stopping chưa dừng trước giới hạn.
>
> Thời gian dự đoán toàn bộ 10.000 ảnh là khoảng 6,7 giây, nhanh hơn rất nhiều so với thời gian huấn luyện. Đây là điều bình thường vì khi dự đoán, model chỉ chạy theo chiều xuôi và không phải cập nhật trọng số.

### Mở và nói về kết quả code

Mở bảng:

```python
results_df.sort_values("Macro F1", ascending=False)
```

> Bảng được sắp xếp theo Macro F1 từ cao xuống thấp. Compact ResNet đứng đầu với Macro F1 bằng 0,5055 và accuracy bằng 0,5062. Trong lần chạy này, ResNet là mô hình tốt nhất trong bốn mô hình được so sánh.

Mở output:

```text
Compact ResNet: params=710,468, best_epoch=20,
test_acc=0.5062, macro_f1=0.5055
```

> Số thập phân 0,5062 tương đương 50,62 phần trăm. Tương tự, 0,5055 tương đương 50,55 phần trăm.

---

## Slide 9 — Biểu đồ và các dự đoán sai

### Nội dung trên slide

- Learning curve: model học như thế nào qua từng epoch
- Confusion matrix: lớp nào dễ bị nhầm với lớp nào
- Ảnh dự đoán sai: xem trực tiếp lỗi của model

### Lời thuyết trình

> Biểu đồ loss và accuracy cho thấy quá trình học của model qua từng epoch. Đường train phản ánh kết quả trên dữ liệu model đang học, còn đường validation phản ánh khả năng áp dụng lên dữ liệu chưa dùng để cập nhật trọng số. Khoảng cách quá lớn giữa hai đường có thể là dấu hiệu overfitting.
>
> Confusion matrix cho biết lớp thật và lớp model dự đoán. Các ô trên đường chéo là dự đoán đúng, còn các ô ngoài đường chéo là các trường hợp nhầm lẫn. Với CIFAR-100, model dễ nhầm các lớp có hình dạng hoặc màu sắc gần giống nhau vì ảnh chỉ có kích thước 32 nhân 32.

### Mở và nói về code

```python
wrong_idx = np.flatnonzero(best_pred != y_test)
```

> Dòng này tìm vị trí của tất cả ảnh mà nhãn dự đoán khác nhãn thật. Phần code phía dưới hiển thị các ảnh sai cùng nhãn thật và nhãn model dự đoán để phân tích lỗi.

---

## Slide 10 — Kết luận

### Nội dung trên slide

- Shortcut là điểm quan trọng nhất của ResNet
- Compact ResNet tốt nhất trong benchmark hiện tại
- Accuracy: 50,62%; Macro F1: 50,55%
- Có thể cải thiện bằng cách train lâu hơn hoặc dùng model sâu hơn

### Lời thuyết trình

> Tóm lại, điểm quan trọng nhất của ResNet là shortcut connection. Nó cho phép giữ lại thông tin cũ và giúp gradient truyền qua mạng dễ hơn. Trong bốn model được thử nghiệm, Compact ResNet đạt Accuracy và Macro F1 cao nhất. Tuy nhiên, best epoch vẫn là epoch 20 nên model có khả năng còn cải thiện nếu tăng số epoch và điều chỉnh thêm learning rate. Đây là Compact ResNet tự xây dựng cho bài tập, không phải phiên bản ResNet-18 hay ResNet-50 đã được huấn luyện sẵn.

---

## Câu mở đầu

> Em xin trình bày phần Compact ResNet dùng để phân loại ảnh CIFAR-100. Em sẽ giải thích ngắn gọn ResNet là gì, code xây dựng và huấn luyện model, sau đó nhận xét kết quả chạy.

## Câu kết thúc

> Phần trình bày Compact ResNet của em đến đây là hết. Điểm chính em rút ra là shortcut giúp model sâu học ổn định hơn và trong thí nghiệm này ResNet cho kết quả tốt nhất.

## Những câu thầy có thể hỏi

### Vì sao gọi là Compact ResNet?

> Vì đây là phiên bản ResNet được rút gọn, chỉ có ba stage và hai residual block trong mỗi stage để phù hợp với ảnh nhỏ 32 nhân 32 và thời gian chạy của bài tập.

### Vì sao đầu ra có 100 neuron?

> Vì CIFAR-100 có 100 lớp. Mỗi neuron đại diện cho xác suất của một lớp.

### Vì sao dùng softmax?

> Softmax biến đầu ra thành 100 xác suất có tổng bằng 1, từ đó có thể chọn lớp có xác suất lớn nhất.

### Vì sao dùng validation và test riêng?

> Validation dùng để điều chỉnh quá trình train. Test chỉ dùng để đánh giá cuối cùng, giúp kết quả khách quan hơn.

### Accuracy 50,62% có thấp không?

> Nếu so với bài toán 10 lớp thì chưa cao, nhưng CIFAR-100 có tới 100 lớp và ảnh rất nhỏ. Đoán ngẫu nhiên chỉ khoảng 1%, nên 50,62% cho thấy model đã học được đặc trưng rõ ràng. Kết quả vẫn có thể cải thiện bằng cấu hình mạnh hơn.

### Vì sao không chạy lại lúc thuyết trình?

> Lần train đã lưu trong notebook mất khoảng 2.018 giây, tức hơn 33 phút. Vì vậy khi thuyết trình chỉ nên mở code và kết quả đã chạy sẵn.
