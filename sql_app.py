import streamlit as st
import pandas as pd
import sqlite3
from io import StringIO
import random
import networkx as nx
import matplotlib.pyplot as plt
import altair as alt
import graphviz
from streamlit_ace import st_ace
from streamlit_lottie import st_lottie
import requests

# Configuration de la page
st.set_page_config(layout="wide", page_title="SQL Trainer", page_icon="🎓")

# Styles CSS personnalisés
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        width: 100%;
    }
    .stSelectbox>div>div>select {
        width: 100%;
    }
    .dataframe {
        width: 100%;
    }
    .css-1d391kg {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour créer une base de données SQLite à partir de fichiers CSV
@st.cache_resource
def create_db_from_csv(csv_files, separator):
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    for file in csv_files:
        content = file.getvalue().decode('utf-8')
        df = pd.read_csv(StringIO(content), sep=separator)
        table_name = file.name.split('.')[0]
        df.to_sql(table_name, conn, index=False, if_exists='replace')
    return conn

# Fonction pour exécuter une requête SQL
def execute_query(conn, query):
    try:
        return pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        st.error(f"Erreur SQL : {e}")
        return None

# Fonction pour charger les animations Lottie
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Fonction pour créer une explication visuelle de la requête
def create_query_visualization(query):
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    
    parts = query.lower().split()
    if 'select' in parts:
        dot.node('select', 'SELECT', shape='box')
        if 'from' in parts:
            dot.node('from', 'FROM', shape='box')
            dot.edge('select', 'from')
            table_index = parts.index('from') + 1
            if table_index < len(parts):
                dot.node('table', parts[table_index], shape='ellipse')
                dot.edge('from', 'table')
    
    if 'where' in parts:
        dot.node('where', 'WHERE', shape='diamond')
        dot.edge('table', 'where')
    
    if 'join' in parts:
        dot.node('join', 'JOIN', shape='box')
        dot.edge('table', 'join')
        join_index = parts.index('join') + 1
        if join_index < len(parts):
            dot.node('join_table', parts[join_index], shape='ellipse')
            dot.edge('join', 'join_table')
    
    return dot

# Fonction pour créer un graphique de progression
def create_progress_chart(completed, total):
    data = pd.DataFrame({
        'category': ['Complétés', 'Restants'],
        'value': [completed, total - completed]
    })
    chart = alt.Chart(data).mark_arc().encode(
        theta='value',
        color=alt.Color('category:N', scale=alt.Scale(domain=['Complétés', 'Restants'], range=['#28a745', '#dc3545'])),
        tooltip=['category', 'value']
    ).properties(width=200, height=200)
    return chart

# Fonction pour générer des exercices dynamiques
def generate_dynamic_exercises(conn):
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    exercises = []

    for table in tables:
        table_name = table[0]
        columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        column_names = [col[1] for col in columns]

        # Niveau Facile (15 exercices)
        exercises.extend([
            {
                "id": len(exercises) + 1,
                "niveau": "Facile",
                "competence": "SELECT",
                "question": f"Sélectionnez toutes les colonnes de la table '{table_name}'",
                "reponse": f"SELECT * FROM {table_name}"
            },
            {
                "id": len(exercises) + 1,
                "niveau": "Facile",
                "competence": "COUNT",
                "question": f"Comptez le nombre total d'enregistrements dans la table '{table_name}'",
                "reponse": f"SELECT COUNT(*) FROM {table_name}"
            },
            {
                "id": len(exercises) + 1,
                "niveau": "Facile",
                "competence": "WHERE",
                "question": f"Sélectionnez toutes les colonnes de '{table_name}' où la première colonne n'est pas NULL",
                "reponse": f"SELECT * FROM {table_name} WHERE {column_names[0]} IS NOT NULL"
            },
            {
                "id": len(exercises) + 1,
                "niveau": "Facile",
                "competence": "ORDER BY",
                "question": f"Sélectionnez toutes les colonnes de '{table_name}' triées par '{column_names[0]}' en ordre décroissant",
                "reponse": f"SELECT * FROM {table_name} ORDER BY {column_names[0]} DESC"
            },
            {
                "id": len(exercises) + 1,
                "niveau": "Facile",
                "competence": "LIMIT",
                "question": f"Sélectionnez les 10 premiers enregistrements de '{table_name}'",
                "reponse": f"SELECT * FROM {table_name} LIMIT 10"
            }
        ])

        # Ajoutez ici plus d'exercices de niveau Facile jusqu'à en avoir 15 au total

        # Niveau Intermédiaire (20 exercices)
        if len(tables) > 1:
            other_table = random.choice([t[0] for t in tables if t[0] != table_name])
            exercises.extend([
                {
                    "id": len(exercises) + 1,
                    "niveau": "Intermédiaire",
                    "competence": "JOIN",
                    "question": f"Joignez les tables '{table_name}' et '{other_table}' sur une colonne commune",
                    "reponse": f"SELECT * FROM {table_name} INNER JOIN {other_table} ON {table_name}.id = {other_table}.{table_name}_id"
                },
                {
                    "id": len(exercises) + 1,
                    "niveau": "Intermédiaire",
                    "competence": "GROUP BY",
                    "question": f"Comptez le nombre d'enregistrements dans '{table_name}' groupés par '{column_names[0]}'",
                    "reponse": f"SELECT {column_names[0]}, COUNT(*) FROM {table_name} GROUP BY {column_names[0]}"
                },
                {
                    "id": len(exercises) + 1,
                    "niveau": "Intermédiaire",
                    "competence": "HAVING",
                    "question": f"Comptez le nombre d'enregistrements dans '{table_name}' groupés par '{column_names[0]}', en ne gardant que les groupes ayant plus de 5 enregistrements",
                    "reponse": f"SELECT {column_names[0]}, COUNT(*) FROM {table_name} GROUP BY {column_names[0]} HAVING COUNT(*) > 5"
                },
                {
                    "id": len(exercises) + 1,
                    "niveau": "Intermédiaire",
                    "competence": "Sous-requête",
                    "question": f"Sélectionnez les enregistrements de '{table_name}' où '{column_names[0]}' est supérieur à la moyenne",
                    "reponse": f"SELECT * FROM {table_name} WHERE {column_names[0]} > (SELECT AVG({column_names[0]}) FROM {table_name})"
                }
            ])

        # Ajoutez ici plus d'exercices de niveau Intermédiaire jusqu'à en avoir 20 au total

        # Niveau Difficile (10 exercices)
        if len(tables) > 2:
            third_table = random.choice([t[0] for t in tables if t[0] not in [table_name, other_table]])
            exercises.extend([
                {
                    "id": len(exercises) + 1,
                    "niveau": "Difficile",
                    "competence": "Jointures multiples",
                    "question": f"Joignez les tables '{table_name}', '{other_table}' et '{third_table}'",
                    "reponse": f"SELECT * FROM {table_name} t1 INNER JOIN {other_table} t2 ON t1.id = t2.{table_name}_id INNER JOIN {third_table} t3 ON t2.id = t3.{other_table}_id"
                },
                {
                    "id": len(exercises) + 1,
                    "niveau": "Difficile",
                    "competence": "Sous-requête corrélée",
                    "question": f"Trouvez les enregistrements dans '{table_name}' qui ont une valeur de '{column_names[0]}' supérieure à la moyenne de leur groupe '{column_names[1]}'",
                    "reponse": f"SELECT * FROM {table_name} t1 WHERE {column_names[0]} > (SELECT AVG({column_names[0]}) FROM {table_name} t2 WHERE t2.{column_names[1]} = t1.{column_names[1]})"
                },
                {
                    "id": len(exercises) + 1,
                    "niveau": "Difficile",
                    "competence": "UNION",
                    "question": f"Unissez les résultats des requêtes sur '{table_name}' et '{other_table}' en sélectionnant une colonne commune",
                    "reponse": f"SELECT {column_names[0]} FROM {table_name} UNION SELECT {column_names[0]} FROM {other_table}"
                }
            ])

        # Ajoutez ici plus d'exercices de niveau Difficile jusqu'à en avoir 10 au total

        # Niveau Expert (5 exercices)
        exercises.extend([
            {
                "id": len(exercises) + 1,
                "niveau": "Expert",
                "competence": "Requête récursive",
                "question": f"Écrivez une requête récursive pour générer une séquence de nombres de 1 à 10",
                "reponse": f"WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM seq WHERE n < 10) SELECT n FROM seq"
            },
            {
                "id": len(exercises) + 1,
                "niveau": "Expert",
                "competence": "Window function",
                "question": f"Calculez une moyenne mobile sur 3 périodes pour la colonne '{column_names[0]}' de la table '{table_name}'",
                "reponse": f"SELECT {column_names[0]}, AVG({column_names[0]}) OVER (ORDER BY {column_names[0]} ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as moving_avg FROM {table_name}"
            }
        ])

        # Ajoutez ici plus d'exercices de niveau Expert jusqu'à en avoir 5 au total

    return exercises

# Sidebar pour le chargement des fichiers CSV et la navigation
st.sidebar.header("Navigation")
uploaded_files = st.sidebar.file_uploader("Choisissez un ou plusieurs fichiers CSV", type=["csv", "txt"], accept_multiple_files=True)

# Ajout du sélecteur de séparateur
separator = st.sidebar.selectbox("Choisissez le séparateur", [",", ";", "\t"], index=0)

if uploaded_files:
    conn = create_db_from_csv(uploaded_files, separator)
    
    # Navigation
    page = st.sidebar.radio("Pages", ["Visualisation et Requêtes", "Exercices", "Schéma de base de données", "Assistant SQL", "Progression"])
    
    if page == "Visualisation et Requêtes":
        st.header("Visualisation et Requêtes SQL")
        
        table_name = st.selectbox("Choisissez une table à visualiser", [file.name.split('.')[0] for file in uploaded_files])
        if table_name:
            df = execute_query(conn, f"SELECT * FROM {table_name} LIMIT 100")
            if df is not None and not df.empty:
                st.dataframe(df)
            else:
                st.warning("Aucune donnée à afficher pour cette table.")
            
            st.subheader("Éditeur SQL")
            query = st_ace(
                placeholder="Entrez votre requête SQL ici",
                language="sql",
                theme="monokai",
                keybinding="vscode",
                font_size=14,
                min_lines=10,
                key="sql_editor"
            )
            
            if st.button("Exécuter la requête"):
                result = execute_query(conn, query)
                if result is not None and not result.empty:
                    st.success("Requête exécutée avec succès!")
                    st.dataframe(result)
                elif result is not None:
                    st.warning("La requête a été exécutée, mais n'a retourné aucun résultat.")
    
    elif page == "Exercices":
        st.header("Exercices SQL")
        
        lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
        st_lottie(lottie_coding, speed=1, height=200, key="initial")

        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox("Difficulté", ["Tous", "Facile", "Intermédiaire", "Difficile", "Expert"])
        with col2:
            skill = st.selectbox("Compétence", ["Tous", "SELECT", "WHERE", "JOIN", "GROUP BY", "HAVING", "Sous-requête", "Window function", "Requête récursive"])
        
        # Générer des exercices dynamiques
        exercises = generate_dynamic_exercises(conn)
        
        filtered_exercises = [ex for ex in exercises if 
                            (difficulty == "Tous" or ex['niveau'] == difficulty) and 
                            (skill == "Tous" or ex['competence'] == skill)]
        
        # Ajout d'une pagination pour gérer un grand nombre d'exercices
        exercises_per_page = 5
        num_pages = max(1, len(filtered_exercises) // exercises_per_page + (1 if len(filtered_exercises) % exercises_per_page > 0 else 0))
        page_number = st.selectbox("Page", range(1, num_pages + 1), key="page_select")
        start_idx = (page_number - 1) * exercises_per_page
        end_idx = min(start_idx + exercises_per_page, len(filtered_exercises))
        
        for i, exercise in enumerate(filtered_exercises[start_idx:end_idx], start=start_idx):
            st.subheader(f"Exercice {exercise['id']} - {exercise['niveau']} - {exercise['competence']}")
            st.markdown(f"**Question:** {exercise['question']}")
            
            if st.button(f"Besoin d'aide pour l'exercice {exercise['id']} ?", key=f"help_button_{i}"):
                st.info("Conseil : Assurez-vous d'utiliser les noms de tables et de colonnes exactement comme ils apparaissent dans votre base de données.")
                st.graphviz_chart(create_query_visualization(exercise['reponse']))
            
            user_query = st_ace(
                placeholder="Votre réponse",
                language="sql",
                theme="monokai",
                keybinding="vscode",
                font_size=14,
                min_lines=5,
                key=f"exercise_{i}"
            )
            
            if st.button(f"Vérifier l'exercice {exercise['id']}", key=f"check_button_{i}"):
                lottie_url = "https://assets3.lottiefiles.com/datafiles/bEYvzB8QfV3EM9a/data.json"
                lottie_json = load_lottieurl(lottie_url)
                st_lottie(lottie_json, speed=1, height=200, key=f"check_{i}")
                
                user_result = execute_query(conn, user_query)
                correct_result = execute_query(conn, exercise['reponse'])
                
                if user_result is not None and correct_result is not None:
                    if user_result.equals(correct_result):
                        st.success("Correct ! 🎉")
                        st.code(user_query, language="sql")
                    else:
                        st.error("Incorrect. Essayez encore.")
                        st.code(user_query, language="sql")
                        
                        st.subheader("Comparaison visuelle")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Votre requête :")
                            st.graphviz_chart(create_query_visualization(user_query))
                        with col2:
                            st.write("Requête correcte :")
                            st.graphviz_chart(create_query_visualization(exercise['reponse']))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Votre résultat:")
                        st.dataframe(user_result)
                    with col2:
                        st.write("Résultat attendu:")
                        st.dataframe(correct_result)
                else:
                    st.error("Une erreur s'est produite lors de l'exécution de la requête. Vérifiez votre syntaxe SQL.")

    elif page == "Schéma de base de données":
        st.header("Schéma de base de données")
        
        G = nx.Graph()
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        
        # Dictionnaire pour stocker les colonnes de chaque table
        table_columns = {}
        
        for table in tables:
            table_name = table[0]
            G.add_node(table_name, node_type='table')
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            table_columns[table_name] = [col[1] for col in columns]
            
            for col in columns:
                col_name = col[1]
                G.add_node(f"{table_name}.{col_name}", node_type='column')
                G.add_edge(table_name, f"{table_name}.{col_name}")
                
                # Vérifier si cette colonne est une clé étrangère potentielle
                if col_name.endswith('_id'):
                    related_table = col_name[:-3]
                    if related_table in [t[0] for t in tables]:
                        G.add_edge(f"{table_name}.{col_name}", related_table, relation='foreign_key')

        # Création du graphique
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.9, iterations=50)
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, 
                               nodelist=[node for node, data in G.nodes(data=True) if data['node_type'] == 'table'])
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=1500, 
                               nodelist=[node for node, data in G.nodes(data=True) if data['node_type'] == 'column'])
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos)
        
        # Ajouter les labels
        labels = {node: node.split('.')[-1] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Ajouter les labels des arêtes pour les clés étrangères
        edge_labels = {(u, v): 'FK' for (u, v, d) in G.edges(data=True) if d.get('relation') == 'foreign_key'}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Schéma de la base de données")
        plt.axis('off')
        st.pyplot(plt)
        
        # Afficher la liste des colonnes pour chaque table
        st.subheader("Détails des tables")
        for table, columns in table_columns.items():
            st.write(f"**{table}**: {', '.join(columns)}")
    
    elif page == "Assistant SQL":
        st.header("Assistant SQL")
        user_question = st.text_input("Posez votre question sur SQL ici:")
        if user_question:
            responses = {
                "select": "La clause SELECT est utilisée pour sélectionner des données d'une base de données. Les données renvoyées sont stockées dans une table de résultats, appelée jeu de résultats.",
                "where": "La clause WHERE est utilisée pour filtrer les enregistrements. Elle est utilisée pour extraire uniquement les enregistrements qui remplissent une condition spécifiée.",
                "join": "Un JOIN est utilisé pour combiner des lignes de deux ou plusieurs tables, en fonction d'une colonne liée entre elles.",
                "group by": "La clause GROUP BY est utilisée pour regrouper des lignes qui ont les mêmes valeurs dans des colonnes spécifiées.",
                "having": "La clause HAVING spécifie une condition de recherche pour un groupe ou un agrégat.",
                "order by": "La clause ORDER BY est utilisée pour trier le jeu de résultats dans un ordre ascendant ou descendant.",
                "insert": "La commande INSERT INTO est utilisée pour insérer de nouveaux enregistrements dans une table.",
                "update": "La commande UPDATE est utilisée pour modifier les enregistrements existants dans une table.",
                "delete": "La commande DELETE est utilisée pour supprimer des enregistrements existants d'une table."
            }
            for key, value in responses.items():
                if key in user_question.lower():
                    st.info(value)
                    break
            else:
                st.warning("Désolé, je n'ai pas de réponse spécifique à cette question. Voici quelques sujets sur lesquels je peux vous aider : SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY, INSERT, UPDATE, DELETE.")

    elif page == "Progression":
        st.header("Votre Progression")
        
        # Ici, vous devriez idéalement avoir un système pour suivre la progression de l'utilisateur
        # Pour cet exemple, nous utiliserons des valeurs statiques
        completed_exercises = 5  # À remplacer par la vraie valeur
        total_exercises = len(generate_dynamic_exercises(conn))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Exercices complétés", f"{completed_exercises}/{total_exercises}")
        with col2:
            progress_percentage = (completed_exercises / total_exercises) * 100
            st.progress(progress_percentage)
        
        st.subheader("Répartition des exercices")
        st.altair_chart(create_progress_chart(completed_exercises, total_exercises))

        # Affichage des derniers exercices complétés
        st.subheader("Derniers exercices complétés")
        # Ici, vous devriez afficher les derniers exercices que l'utilisateur a complétés
        # Pour cet exemple, nous utiliserons des données factices
        last_completed = [
            {"id": 1, "niveau": "Facile", "competence": "SELECT"},
            {"id": 3, "niveau": "Intermédiaire", "competence": "WHERE"},
            {"id": 5, "niveau": "Difficile", "competence": "JOIN"}
        ]
        for exercise in last_completed:
            st.write(f"Exercice {exercise['id']} - {exercise['niveau']} - {exercise['competence']}")

else:
    st.error("Veuillez charger un ou plusieurs fichiers CSV pour commencer.")
    st.header("Comment commencer ?")
    st.markdown("""
    1. Utilisez le sélecteur de fichiers dans la barre latérale pour charger vos fichiers CSV.
    2. Choisissez le séparateur approprié pour vos fichiers (virgule, point-virgule ou tabulation).
    3. Une fois les fichiers chargés, naviguez entre les différentes pages pour explorer vos données, pratiquer des requêtes SQL et suivre votre progression.
    4. N'hésitez pas à utiliser l'assistant SQL si vous avez des questions !
    """)

    # Affichage d'une animation ou d'une image pour rendre la page d'accueil plus attrayante
    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
    st_lottie(lottie_hello, speed=1, height=400, key="hello")