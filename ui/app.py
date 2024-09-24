"""這段代碼包括以下步驟：

Flask 應用設置：初始化應用並設置確保所需目錄存在。
首頁路由：定義首頁路由，返回主頁面模板。
生成圖片路由：
獲取上傳的圖片和選擇的模型。
清空目錄以存儲新的圖片。
保存上傳的圖片到相應的目錄。
設置環境變量以指定使用的 CUDA 設備。
根據選擇的模型設置命令參數並組裝命令。
執行命令並檢查是否成功，若失敗則重定向到錯誤頁面。
結果頁面路由：顯示生成的圖片。
錯誤頁面路由：顯示錯誤信息。
運行應用：以 debug 模式運行 Flask 應用。

Returns:
    _type_: _description_
"""

from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
import shutil

app = Flask(__name__)

# 確保目錄存在，若不存在則建立
src_dir = 'ui/data/src/t'
ref_dir = 'ui/data/ref/t'
res_dir = './ui/static/res/'
os.makedirs(src_dir, exist_ok=True)
os.makedirs(ref_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

@app.route('/')
def index():
    # 返回主頁面
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # 獲取上傳的圖片和模型選擇
    source_images = request.files.getlist('source_image')
    reference_images = request.files.getlist('reference_image')
    model = request.form.get('model')

    # 清空指定目錄中的所有文件
    def clear_directory(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    clear_directory(src_dir)
    clear_directory(ref_dir)

    # 保存上傳的圖片到 src 和 ref 目錄
    for source_image in source_images:
        if source_image.filename:
            source_image_path = os.path.join(src_dir, source_image.filename)
            source_image.save(source_image_path)

    for reference_image in reference_images:
        if reference_image.filename:
            reference_image_path = os.path.join(ref_dir, reference_image.filename)
            reference_image.save(reference_image_path)

    # 設置環境變量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'

    # 根據選擇的模型設置命令參數
    model_checkpoints = {
        'resnet': 'expr/checkpoints/celeba_hq',
        'swinnet': 'expr/checkpoints/swinnet',
        'anime': 'expr/checkpoints/faces',
        'animal': 'expr/checkpoints/afhq',
        'swinunet': "expr/checkpoints/swinunet"
    }

    # 模型骨架設置
    bockbone = {
        'resnet': 'ResNet',
        'swinnet': 'SwinStyle',
        'anime': 'ResNet',
        'animal': 'ResNet',
        'swinunet': "SwinUnet"
    }

    # 風格維度設置
    style_dim = {
        'resnet': '64',
        'swinnet': '256',
        'anime': '64',
        'animal': '64',
        'swinunet': '128'
    }

    # 高通濾波設置
    w_hpf = '0' if model == 'animal' else '1'

    # 組裝命令
    command = [
        'python', 'main.py', '--mode', 'sample', '--num_domains', '1',
        '--resume_iter', '100000', '--generator_backbone',
        bockbone[model], '--w_hpf', w_hpf, '--checkpoint_dir', model_checkpoints[model],
        '--style_dim', style_dim[model],
        '--result_dir', res_dir, '--src_dir', 'ui/data/src/', '--ref_dir', 'ui/data/ref/'
    ]

    # 打印並執行命令
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 檢查命令執行是否成功
    if result.returncode != 0:
        error_message = result.stderr
        return redirect(url_for('error', message=error_message))

    return redirect(url_for('result'))

@app.route('/result')
def result():
    # 顯示生成的圖片
    generated_images = os.listdir(res_dir)
    return render_template('result.html', images=generated_images)

@app.route('/error')
def error():
    # 顯示錯誤信息
    message = request.args.get('message', '')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    # 運行 Flask 應用
    app.run(debug=True)
