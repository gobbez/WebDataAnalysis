from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import os
import base64

# Supervised Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestRegressor
# Boosts Models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
# Train / Test and Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
# UnSupervised Models
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import plot_tree


# Avvia Flask
app = Flask(__name__)

# Global variable to store DataFrame
df = None

def percorso_file(cartella, nome_file):
    """
    Crea il percorso dove è contenuto il file (serve in modo che prende il percorso anche se si sposta la cartella del progetto)
    :parametri:
        cartella: (str) il nome della cartella dove inviare o salvare il file
        nome_file: (str) il nome del file da inviare o salvare, scrivere anche l'estensione del file (.jpg, .csv, ecc)
    :return: file_path: il percorso del file
    """
    # Ottieni il percorso assoluto della directory corrente (dove si trova il tuo script Python)
    current_dir = os.path.dirname(__file__)
    # Costruisci il percorso assoluto del file
    file_path = os.path.join(current_dir, cartella, nome_file)
    return file_path


def create_graph(columns, type):
    if len(columns) < 2:
        # Show an empty graph with invalid columns
        plt.plot(0, 0)
        plt.title('There must be at least 2 columns')
    else:
        x = df[columns[0]]
        y = df[columns[1]]
        print(x.dtype, y.dtype)
        if x.dtype not in ('int', 'float') or y.dtype not in ('int', 'float'):
            # Show an empty graph with invalid type
            plt.plot(0,0)
            plt.title('Invalid data type')
        else:
            plt.figure(figsize=(6,5))
            if type == 'plot':
                plt.plot(df[columns[0]], df[columns[1]])
            elif type == 'scatter':
                plt.scatter(df[columns[0]], df[columns[1]])
            elif type == 'bar':
                plt.bar(df[columns[0]], df[columns[1]])
            elif type == 'hist':
                plt.hist(df[columns[0]])
                plt.title(f'{type.upper()} of {columns[0]}')
            plt.xlabel(columns[0])
            plt.xticks(rotation=90)
            plt.ylabel(columns[1])
            if type != 'hist':
                plt.title(f'{type.upper()} of {columns[0]} vs {columns[1]}')
    # Salva il grafico in un oggetto BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    # Converti l'immagine in una stringa base64
    graph_url = base64.b64encode(img.getvalue()).decode()
    return f'data:image/png;base64,{graph_url}'


@app.route('/update_graph', methods=['POST'])
def update_graph():
    global df
    x_col = request.form.get('x_column')
    y_col = request.form.get('y_column')
    columns = [x_col, y_col]
    graph_type = request.form.get('graph')

    if x_col and y_col and graph_type:
        graph = create_graph(columns, graph_type)
        return jsonify({'graph': graph})
    return jsonify({'error': 'Invalid data'})


@app.route('/update_data', methods=['POST'])
def update_data():
    global df
    columns = request.form.getlist('columns[]')
    if columns:
        data = df[columns]
        table = data.to_html(classes='data')
        return jsonify({'table': table})
    return jsonify({'error': 'No columns selected'})



def create_machine_learning(columns, model):
    try:
        # Check if there are at least 2 columns
        if len(columns) < 2:
            accuracy = 0
            results = 'No results shown'
            # Show an empty graph with invalid columns
            plt.plot(0, 0)
            plt.title('There must be at least 2 columns')
        else:
            if model == 'all classifiers':
                best_classifier = None
                best_accuracy = 0
                best_ypred = 0
                SEED = 42
                # X contains the features that we want to use, y is the feature we want to calculate
                y = df[columns[-1]].values
                X = df[columns].values

                # Split the dataset on train and test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

                # Instantiate Classifiers Modules
                lr = LogisticRegression(random_state=SEED)
                knn = KNeighborsClassifier(n_neighbors=7)
                dt = DecisionTreeClassifier(min_samples_leaf=0.10, random_state=SEED)
                ada = AdaBoostClassifier(estimator=dt, n_estimators=200, random_state=SEED)

                # Create a list of classifiers
                estimators = [('lr', lr), ('knn', knn), ('dt', dt), ('ada', ada)]
                # Create the VotingClassifier with the list
                vc = VotingClassifier(estimators=estimators)

                # Define the list of classifiers
                classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt),
                               ('Voting Classifier', vc), ('Ada Boost', ada)]

                # Iterate over the list of classifiers
                results = []
                for clf_name, clf in classifiers:
                    # Fit clf to the training set
                    clf.fit(X_train, y_train)

                    # Predict y_pred
                    y_pred = clf.predict(X_test)

                    # Calculate accuracy and append result
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append((clf_name, str(round(accuracy*100))+'%'))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_classifier = clf_name
                        best_ypred = y_pred

                # Create a scatter plot with dt predictions and real values
                plt.figure(figsize=(6, 5))
                sns.scatterplot(x=X_test[:, 0], y=y_test, label='Actual', s=80, alpha=1)
                sns.scatterplot(x=X_test[:, 0], y=best_ypred, label='Predicted', s=160, alpha=0.2)
                plt.xlabel(columns[0])
                plt.ylabel(columns[-1])
                plt.legend()
                plt.title(f'Predictions by Best Predictor ({best_classifier})')
    except:
        # If exception is raised means that 1 or more columns aren't of proper type
        accuracy = 0
        results = 'No results shown'
        # Show an empty graph with invalid columns
        plt.plot(0, 0)
        plt.title('One or more column(s) type are invalid')
    # Salva il grafico in un oggetto BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    # Converti l'immagine in una stringa base64
    graph_url = base64.b64encode(img.getvalue()).decode()
    return round(accuracy*100), results, f'data:image/png;base64,{graph_url}'


@app.route('/update_ml', methods=['POST'])
def update_ml():
    global df
    ml_model = request.form.get('ml_model')
    columns = request.form.getlist('columns[]')
    accuracy, results, ml_graph = create_machine_learning(columns, ml_model)
    return jsonify({'ml_acc': accuracy, 'ml_results': results, 'ml_graph': ml_graph})



@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    global df
    if request.method == 'POST':
        print(request.files['file'])
        if not request.files['file']:
            return 'No file selected'
        file = request.files['file']
        # get file extension
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension in ['xls', 'xlsx']:
            file_content = file.read()
            df = pd.read_excel(BytesIO(file_content))
        elif file_extension == 'csv':
            file_content = file.read()
            df = pd.read_csv(BytesIO(file_content))
        else:
            return 'Invalid file format'
        print(f"File uploaded and DataFrame created, {len(df)} rows")
        return redirect(url_for('analysis'))
    return render_template('index.html')


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    global df
    columns = None
    type_graph = None
    type_ml = None
    if df is None:
        return redirect(url_for('home'))

    if request.method == 'POST':
        columns = request.form.getlist('columns')
        type_graph = request.form.get('graph')
        type_ml = request.form.get('ml')
        graph = create_graph(columns, type_graph)
        ml_acc, ml_results, ml_graph = create_machine_learning(columns, type_ml)
        ml_acc = round(ml_acc*100)
        if columns:
            data = df[columns]
            return render_template('analysis.html', df=df, tables=[data.to_html(classes='data')], titles=[''], graph=graph, ml_acc=ml_acc, ml_results=ml_results, ml_graph=ml_graph)
        else:
            msg_columns = 'No columns selected'
            return render_template('analysis.html', df=df)

    return render_template('analysis.html', df=df)



# Home
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)