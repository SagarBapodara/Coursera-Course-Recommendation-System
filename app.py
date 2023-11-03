from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask,request,redirect,render_template
import pickle
app=Flask(__name__)
with open("./models/combined_df.pkl","rb") as f:
    combined_df=pickle.load(f)
courses = combined_df[['Course Name', 'Course Description']].to_dict(orient='records')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform([course['Course Description'] for course in courses])

def get_recommendations(user_description, n=5):
    recommended_courses = []
    course_title = ''
    user_vector = tfidf.transform([user_description])
    cosine_sim = linear_kernel(user_vector, tfidf_matrix)
    for idx in cosine_sim.argsort()[0][::-1][:n]:
        course_title = courses[idx]['Course Name']
        recommended_courses.append(course_title)
    filtered_df = combined_df[combined_df['Course Name'].isin(recommended_courses)]
    return filtered_df

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        des = request.form.get("des")
        num = request.form.get("num")
        if not num:
            num = 5
        recommendations = get_recommendations(des,int(num))
        params={"title":"Home", "result":{'active':True,"data" : recommendations}}
        return render_template("index.html", params=params)
    params={"title":"Home","result":{'active':False}}
    return render_template("index.html", params=params)
if __name__ == "__main__":
    app.run(debug=True)
