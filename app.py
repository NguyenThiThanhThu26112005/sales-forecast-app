from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib

# Cấu hình Matplotlib để chạy ở chế độ "headless" (không cần GUI)
# Quan trọng cho việc deploy trên server
matplotlib.use('Agg')

app = Flask(__name__)

def safe_mape(y_true, y_pred):
    """
    Tính toán Mean Absolute Percentage Error (MAPE) một cách an toàn,
    tránh lỗi chia cho 0.
    """
    y_true = pd.Series(y_true).replace(0, np.nan)
    y_pred = pd.Series(y_pred)
    
    # Đảm bảo y_true và y_pred có cùng chỉ số sau khi bỏ qua NaN
    mask = y_true.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if y_true.empty:
        return 0.0  # Hoặc np.nan tùy vào logic bạn muốn

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    result = None
    plot_png = None
    method_name = None

    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        method = request.form.get('method', 'lr')
        
        if not uploaded_file or uploaded_file.filename == '':
            error = 'Không có file được gửi lên.'
            return render_template('index.html', error=error)

        try:
            content = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            error = f'Không thể đọc file CSV: {e}'
            return render_template('index.html', error=error)

        required_features = ['promo', 'temp', 'month_sin', 'month_cos']
        target = 'sales'
        
        if not all(col in df.columns for col in required_features) or target not in df.columns:
            error = f"File thiếu cột cần thiết. Cần: {required_features + [target]}. Có: {list(df.columns)}"
            return render_template('index.html', error=error)

        try:
            X = df[required_features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            if method == 'lr':
                model = LinearRegression()
                method_name = 'Linear Regression'
            else:
                model = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
                method_name = 'Random Forest'

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = safe_mape(y_test.reset_index(drop=True), y_pred)

            # Dự báo minh họa Tháng 13
            new_data = pd.DataFrame({'promo': [0], 'temp': [25.0], 'month_sin': [0.5], 'month_cos': [0.866]})
            forecast = float(model.predict(new_data)[0])

            # Vẽ biểu đồ ra buffer và mã hóa base64
            plt.figure(figsize=(10, 4))
            test_index = X_test.index
            plt.plot(test_index, y_test.values, label='Thực tế (Test)', marker='o')
            plt.plot(test_index, y_pred, label='Dự báo', linestyle='--')
            plt.title(f'Hiệu suất Dự báo - {method_name}')
            plt.xlabel('Chỉ số thời gian')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close() # Đóng biểu đồ để giải phóng bộ nhớ
            buf.seek(0)
            plot_png = base64.b64encode(buf.read()).decode('utf-8')

            result = {
                'r2': f"{r2:.4f}",
                'rmse': f"{rmse:.2f}",
                'mape': f"{mape:.2f}%",
                'forecast': f"{forecast:.0f}"
            }

        except Exception as e:
            error = f'Đã xảy ra lỗi trong quá trình xử lý: {e}'

    # Khi GET request hoặc xử lý POST hoàn tất
    return render_template('index.html', error=error, result=result, plot_png=plot_png, method_name=method_name)


if __name__ == '__main__':
    app.run(debug=True, port=5000)