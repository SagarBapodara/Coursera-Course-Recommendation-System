from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
# Contributed Code
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

# Sagar Bapodara Code 
courses_list = pickle.load(open('./models/courses.pkl','rb'))
similarity = pickle.load(open('./models/similarity.pkl','rb'))

def recommend(course):
    index = courses_list[courses_list['course_name'] == course].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_course_names = []
    for i in distances[1:7]:
        course_name = courses_list.iloc[i[0]].course_name
        recommended_course_names.append(course_name)

    return recommended_course_names

# Routings
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


@app.route('/SagarBapodara',methods=['GET','POST'])
def main():
    course_list = courses_list['course_name'].values
    if request.method == "POST":
        des = request.form.get("des")
        recommended_course_names = recommend(des)
        params={"title":"Home", "result":{'active':True,"data" : recommended_course_names},"course_list":course_list}
        return render_template('sagar.html',params=params)
    params = {"titile":"Sagar Bapodara's Project", "result":{'active':False}, "course_list":course_list}
    return render_template('sagar.html',params=params)


if __name__ == "__main__":
    app.run(debug=True)