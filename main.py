import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("recommendations.csv")

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df["user_interest"])

while True:
    print("\n========== Smart Recommendation System ==========")
    print("1. View Dataset")
    print("2. Get Recommendation")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        print(df)

    elif choice == "2":
        interest = input("Enter your interest: ")

        user_vector = vectorizer.transform([interest])

        similarity = cosine_similarity(user_vector, vectors)

        index = similarity.argmax()

        print("\nRecommended for you:")
        print(df.iloc[index]["recommendation"])

        print("Similarity Score:",
              round(similarity.max()*100,2), "%")

    elif choice == "3":
        print("Thank you!")
        break

    else:
        print("Invalid Choice!")