
from flask import Flask, render_template,request, redirect, url_for, flash
from topkwords import predict
from score import get_value,get_value_regress

app = Flask(__name__)
app.config['SECRET_KEY'] = "andrew"



def topK(prompt,k=None):

    top_K_words_list = predict(prompt,k)
    print("top_K_words_list: ",top_K_words_list)
    temp = []
    for word,points in top_K_words_list: 
        if word not in ["</s>", "<s>"]:
            if word[0] == "Ġ":
                temp.append(word[1::])
            else:
                temp.append(word)
    k= len(temp)
    print("temp: ",temp)
    top_K_words = {'1':temp[:k//3:],'2':temp[k//3:2*(k//3):],'3':temp[2*(k//3)::]}
    return top_K_words

def get_score(prompt):
    return 0


@app.route('/') 
@app.route('/home',methods=['GET','POST']) 
def home():  
    score=0 
    regress_score=0
    top_K_words=[]
    prompt=""
    if 'submit_button' in request.form:
        print("Button pressed")
        prompt = request.form.get("prompt")
        if prompt!="":
            print("Prompt: ", prompt)
            k=None
            top_K_words = topK(prompt,k)
            
            # print(top_K_words_list)
            print("top:", top_K_words)
            
            score = get_value(prompt)
            regress_score = get_value_regress(prompt)
            return render_template('home.html',score=score, regress_score=regress_score ,top_K_words=top_K_words, prompt=prompt)
        else:   
            flash("Please provide the prompt to check first!","danger")
            return redirect(url_for('home'))
    return render_template('home.html',score=score, regress_score=regress_score,top_K_words=top_K_words, prompt=prompt)


    

if(__name__=="__main__"): 
    app.run(debug=True)