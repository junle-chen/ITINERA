from paddleocr import PaddleOCR

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
) 

# 运行 OCR
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)

# 提取并合并所有文本
all_texts_list = []
if result:
    for res in result:
        # res.print() 
        print("--- 单个结果 ---")
        print(res)
        print("---------------")
        # 提取 'rec_texts'
        from paddleocr import PaddleOCR

# 初始化 PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
) 

# 运行 OCR
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)


all_texts_list = []

# 'result' 是一个列表, 包含一个或多个 OCRResult 对象
if result:
    for res in result:
        # 'res' 对象本身就包含 'rec_texts' 属性
        # 我们用 hasattr 来检查它是否存在
        
        if hasattr(res, 'rec_texts') and res.rec_texts:
            # 如果存在, 就用 res.rec_texts 来访问它
            all_texts_list.extend(res.rec_texts)
        else:
            # 否则, 打印一个提示
            print("未能在此结果中找到 'rec_texts'。")

# 将所有文本合并成一个字符串，用空格分隔
combined_text = " ".join(all_texts_list)

print("--- 合并后的文本 ---")
print(combined_text)
