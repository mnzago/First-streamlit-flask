import streamlit as st
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Application de fraude par carte de crédit")
    st.subheader("Auteur: Meri-Nut ZAGO")
    
    #Fonction d'importation des données
    @st.cache(persist=True) #permet de ne pas utiliser trop de mémoire
    def load_data():
        data = pd.read_csv('creditcard.csv')
        return data
    
    #Affiche de la table de données
    df = load_data()
    df_sample = df.sample(100) #On prend que 100 individus pour ne pas alourdir le tps de traitrement
    # st.write(df_sample) # affiche l'ensemble sur toute la page
    # Choix d'afficher ou non les données
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeu de données 'credit card': Echantillon de 100 observations")
        st.write(df_sample)
       
    seed = 123
    
    # Creation d'un train/test set
    #@st.cache(persist=True)
    #def split(df):
    y = df['Class']
    X = df.drop('Class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2, 
            stratify=y, 
            random_state=seed
        )
        
    #X_train, X_test, y_train, y_test = split(df)
    # Mettre des labels plus parlant aux graphiques
    class_names = ['T_Autentique','T_Frauduleuse']
    
    classifier = st.sidebar.selectbox("Choix du Classifier",
                                      ("Random Forest", "SVM", "Logistic Regression")
                                     )
    
    # Analyse de la performance des modèles:
    def plot_perf(graphes):
        if 'Confusion matrix' in graphes:
            st.subheader('Matrice de confusion')
            ConfusionMatrixDisplay.from_estimator(
                model,
                X_test,
                y_test
                #pos_label = class_names
            )
            st.pyplot()
    
        if 'ROC' in graphes:
            st.subheader('Courbe ROC')
            RocCurveDisplay.from_estimator(
                model,
                X_test,
                y_test
            )
            st.pyplot()
            
        if 'Precision-Recall curve' in graphes:
            st.subheader('Courbe precision recall')
            PrecisionRecallDisplay.from_estimator(
                model,
                X_test,
                y_test
            )
            st.pyplot()
            
    # Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyper paramètres du modèle")
        nb_arbres = st.sidebar.number_input(
        "Choisir le nombre d'arbres dans la forêt",
        100, 1000, step=10
        )
        nb_profondeur_arbres = st.sidebar.number_input(
            "Profondeur maximal de l'arbre",
            1, 20, step=1
        )
        bootstrap = st.sidebar.radio(
            "Echantillons bootstrap lors de la création de l'arbre?",
            ("True", "False")
        )
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle",
            ("Confusion matrix", "ROC", "Precision-Recall curve")
        )
        
        # Bouton d'execution 
        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Random Forest Results")
            
            # Initialisation d'un object random forest classifier
            model = RandomForestClassifier(
                n_estimators = nb_arbres,
                max_depth = nb_profondeur_arbres,
                bootstrap=bootstrap
            )
            
            # Entrainement de l'algorithme
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Métriques de performance
            accuracy = model.score(X_test, y_test).round(3)
            precision = precision_score(y_test, y_pred, labels = class_names).round(3)
            recall = recall_score(y_test, y_pred, labels = class_names).round(3)
            
            # Affiche les métriques dans l'application
            st.write("Accuracy:", accuracy)
            st.write("precision:", precision)
            st.write("recall:", recall)
            
            # Afficher les graphiques de performances
            plot_perf(graphes_perf)
            
    # SVM
    if classifier == "SVM":
        st.sidebar.subheader("Hyper paramètres du modèle")
        hyp_c = st.sidebar.number_input(
            "Choisir le paramètre de régularisation",
            0.01, 10.0, step=10.0
        )
        kernel = st.sidebar.radio(
            "Choisir le kernel",
            ("rbf","linear")
        )
        gamma = st.sidebar.radio(
            "Gamma",
            ("scale","auto")
        )
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle",
            ("Confusion matrix", "ROC", "Precision-Recall curve")
        )
        
        # Bouton d'execution 
        if st.sidebar.button("Execution", key="classify"):
            st.subheader("SVM Results")
            
            # Initialisation d'un object random forest classifier
            model = SVC(
                C=hyp_c,
                kernel=kernel,
                gamma=gamma
            )
            
            # Entrainement de l'algorithme
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Métriques de performance
            accuracy = model.score(X_test, y_test).round(3)
            precision = precision_score(y_test, y_pred, labels = class_names).round(3)
            recall = recall_score(y_test, y_pred, labels = class_names).round(3)
            
            # Affiche les métriques dans l'application
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            
            # Afficher les graphiques de performances
            plot_perf(graphes_perf)
            
    # Logistic Regression
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Hyper paramètres du modèle")
        hyp_c = st.sidebar.number_input(
            "Choisir le paramètre de régularisation",
            0.01, 10.0, step=10.0
        )
        nb_max_iter = st.sidebar.number_input(
            "Nombre maximal d'itération",
            100, 1000, step=10
        )
        
        graphes_perf = st.sidebar.multiselect(
            "Choisir un graphique de performance du modèle",
            ("Confusion matrix", "ROC", "Precision-Recall curve")
        )
        
        # Bouton d'execution 
        if st.sidebar.button("Execution", key="classify"):
            st.subheader("Logistic Regression Results")
            
            # Initialisation d'un object logistic regression
            model = LogisticRegression(
                C = hyp_c,
                max_iter = nb_max_iter,
                random_state=seed
            )
            
            # Entrainement de l'algorithme
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Métriques de performance
            accuracy = model.score(X_test, y_test).round(3)
            precision = precision_score(y_test, y_pred).round(3)
            recall = recall_score(y_test, y_pred).round(3)
            
            # Affiche les métriques dans l'application
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)
            
            # Afficher les graphiques de performances
            plot_perf(graphes_perf)
        
if __name__ == '__main__':
    main()