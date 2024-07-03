from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
from model import train_number_model, load_model, predict_next_numbers, reset_model, evaluate_model
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATA_FILE = 'roulette_data.csv'
BETS_FILE = 'bets_data.csv'
RECOMMENDATIONS_FILE = 'recommendations.txt'

def check_and_create_file(file_path, columns):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)

def save_recommendations(recommendations):
    with open(RECOMMENDATIONS_FILE, 'w') as f:
        for item in recommendations:
            f.write("%s\n" % item)

def load_recommendations():
    if not os.path.exists(RECOMMENDATIONS_FILE):
        return []
    with open(RECOMMENDATIONS_FILE, 'r') as f:
        recommendations = [int(line.strip()) for line in f]
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    check_and_create_file(DATA_FILE, ['Number'])
    check_and_create_file(BETS_FILE, ['Bet', 'Result', 'Amount', 'Net_Gain'])

    if request.method == 'POST':
        number = request.form.get('number')
        if number:
            try:
                number = int(number)
                if 0 <= number <= 36:
                    data = pd.read_csv(DATA_FILE)
                    new_row = pd.DataFrame({'Number': [number]})
                    data = pd.concat([data, new_row], ignore_index=True)
                    data.to_csv(DATA_FILE, index=False)
                    flash(f'Number {number} added successfully!', 'success')

                    model = load_model()
                    min_numbers = 20

                    if len(data) >= min_numbers and model:
                        previous_recommendations = load_recommendations()
                        if previous_recommendations:
                            if number in previous_recommendations:
                                result = 'Win'
                                net_gain = 36 * int(session.get('bet_amount', 1)) - 4 * int(session.get('bet_amount', 1))
                            else:
                                result = 'Loss'
                                net_gain = -4 * int(session.get('bet_amount', 1))

                            bet_row = pd.DataFrame({
                                'Bet': [number],
                                'Result': [result],
                                'Amount': [session.get('bet_amount', 1)],
                                'Net_Gain': [net_gain]
                            })

                            bets_data = pd.read_csv(BETS_FILE)
                            bets_data = pd.concat([bets_data, bet_row], ignore_index=True)
                            bets_data.to_csv(BETS_FILE, index=False)

                        recommendations = predict_next_numbers(model, data['Number'].tolist(), n_steps=10, num_predictions=4)
                        save_recommendations(recommendations)
                    else:
                        flash('Not enough data to make recommendations', 'danger')

                else:
                    flash('Number must be between 0 and 36', 'danger')
            except ValueError:
                flash('Please enter a valid number', 'danger')
        return redirect(url_for('index'))

    data = pd.read_csv(DATA_FILE)
    model = load_model()
    min_numbers = 20

    if len(data) >= min_numbers and model:
        recommendations = sorted(predict_next_numbers(model, data['Number'].tolist(), n_steps=10, num_predictions=4))
    else:
        recommendations = ["Not enough data to make recommendations"] if len(data) < min_numbers else ["Model not trained yet"]

    plot_url = generate_plot(data['Number'].tolist())
    gains_losses_plot_url = generate_gains_losses_plot()
    probabilities_plot_url = generate_probabilities_plot(model, data['Number'].tolist(), n_steps=10)
    mean_intervals = calculate_mean_interval(data['Number'].tolist())

    top_numbers = data['Number'].value_counts().head(5).items()

    bets_data = pd.read_csv(BETS_FILE)
    total_bets = len(bets_data)
    total_winnings = bets_data[bets_data['Result'] == 'Win']['Net_Gain'].sum()
    total_losses = bets_data[bets_data['Result'] == 'Loss']['Net_Gain'].sum()

    return render_template(
        'index.html',
        numbers=data['Number'].tolist(),
        recommendations=recommendations,
        plot_url=plot_url,
        gains_losses_plot_url=gains_losses_plot_url,
        probabilities_plot_url=probabilities_plot_url,
        bets_data=bets_data.to_dict(orient='records'),
        total_bets=total_bets,
        total_winnings=total_winnings,
        total_losses=total_losses,
        bet_amount=session.get('bet_amount', 1),
        top_numbers=top_numbers,
        mean_intervals=mean_intervals
    )

@app.route('/train', methods=['POST'])
def train():
    data = pd.read_csv(DATA_FILE)
    numbers = data['Number'].tolist()

    train_number_model(numbers)
    model = load_model()

    train_X, test_X, train_y, test_y = train_test_split(
        np.array([numbers[i:i+10] for i in range(len(numbers)-10)]),
        np.array(numbers[10:]),
        test_size=0.2,
        random_state=42
    )
    accuracy = evaluate_model(model, train_X, train_y)
    flash(f'Model trained successfully with accuracy: {accuracy:.2f}', 'success')

    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    reset_model()
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(BETS_FILE):
        os.remove(BETS_FILE)
    if os.path.exists(RECOMMENDATIONS_FILE):
        os.remove(RECOMMENDATIONS_FILE)
    session.pop('bet_amount', None)
    flash('Model and data reset successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/set_bet_amount', methods=['POST'])
def set_bet_amount():
    bet_amount = request.form.get('bet_amount')
    if bet_amount and bet_amount.isdigit() and int(bet_amount) > 0:
        session['bet_amount'] = int(bet_amount)
        flash(f'Bet amount set to {bet_amount}', 'success')
    else:
        flash('Please enter a valid bet amount', 'danger')
    return redirect(url_for('index'))

def generate_plot(numbers):
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(numbers, bins=range(0, 38), edgecolor='black')
    ax.set_title('Number Frequency')
    ax.set_xlabel('Number')
    ax.set_ylabel('Frequency')

    for count, bin, patch in zip(counts, bins, patches):
        height = patch.get_height()
        ax.annotate(f'{int(count)}', xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def generate_gains_losses_plot():
    bets_data = pd.read_csv(BETS_FILE)
    if bets_data.empty:
        return ""

    fig, ax = plt.subplots()
    bets_data['Cumulative_Gain'] = bets_data['Net_Gain'].cumsum()
    ax.plot(bets_data.index, bets_data['Cumulative_Gain'], marker='o')
    ax.set_title('Cumulative Gains and Losses')
    ax.set_xlabel('Bet Number')
    ax.set_ylabel('Cumulative Net Gain')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def generate_probabilities_plot(model, numbers, n_steps):
    if model is None or len(numbers) < n_steps:
        return ""

    probabilities = []
    for i in range(n_steps, len(numbers)):
        input_seq = numbers[i-n_steps:i]
        input_seq = np.array(input_seq).reshape((1, n_steps, 1))
        pred = model.predict(input_seq, verbose=0)[0][0]
        probabilities.append(pred)

    fig, ax = plt.subplots()
    ax.plot(range(n_steps, len(numbers)), probabilities, marker='o')
    ax.set_title('Predicted Probabilities')
    ax.set_xlabel('Number Index')
    ax.set_ylabel('Probability')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def calculate_mean_interval(numbers):
    intervals = {}
    for number in set(numbers):
        indices = [i for i, x in enumerate(numbers) if x == number]
        if len(indices) > 1:
            intervals[number] = np.mean(np.diff(indices))
        else:
            intervals[number] = None
    return intervals

if __name__ == "__main__":
    app.run(debug=True)
