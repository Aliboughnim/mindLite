from django.shortcuts import render
from django.shortcuts import redirect,get_object_or_404 
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.contrib.auth import authenticate, login,logout
from .models import *
from django.http import HttpResponseRedirect
from django.db.models import Q,Count
from .utils import get_teachers_for_parent
from itertools import zip_longest
from django.http import JsonResponse,HttpResponseBadRequest
import requests
import torch
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from diffusers import StableDiffusionImg2ImgPipeline
import base64
from datetime import date
import os
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.template import loader
import json
from gtts import gTTS
import spacy

from django.shortcuts import render, redirect

from django.templatetags.static import static

from all.my_spacy import add_space_between_number_and_word, additionner_nombres, convert_number_to_digits, create_dependency_graph, detecter_nombre_en_lettres, draw_dependency_graph
import all.spacy_2
from collections import defaultdict
from .models import Course
from django.http import JsonResponse
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pke
import string
import transformers
from flashtext import KeywordProcessor
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('punkt')
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

from django.utils import timezone
nltk.download('omw-1.4')

# Create your views here.
def home(request):
    return render(request,'all/home.html')

def my_login(request):
    if request.method == 'POST':
        us = request.POST['username']
        pwd = request.POST['password']
        user = authenticate(request, username=us, password=pwd)
        print(user)
        if user is not None:
            login(request,user)
            if user.role == User.Role.STUDENT:
                return redirect('s_home')
            elif user.role == User.Role.TEACHER:
                return redirect('t_home') 
            elif user.role == User.Role.PARENT:
                return redirect('p_home')
        else:
            # Handle invalid login
            return render(request, 'all/login.html', {'error': 'Invalid username or password'})
    else:
        return render(request, 'all/login.html')
    
def contact(request):
    return render(request,'all/contact.html')

@login_required(login_url='my_login')
def p_home(request):
    parent = request.user
    stuprof = StudentProfile.objects.filter(parent_user_id=parent.id)
    students = [profile.user for profile in stuprof]
    student = students[0]
    student_id = student.id
    xp_data = Xp.objects.filter(Student_id=student_id).order_by('date')
    xp_values = [xp.xp for xp in xp_data]
    dates = [xp.date for xp in xp_data]

    # Create the plot
    plt.plot(dates, xp_values)
    plt.xlabel('Date')
    plt.ylabel('XP')
    plt.title('{} Results'.format(student.username))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()

    context = {
        'parent': parent,
        'image_base64': image_base64
    }
    return render(request,'all/parent_home.html',context)

@login_required(login_url='my_login')
def s_home(request):
    student = request.user
    student_profile = student.studentprofile
    class_id = student_profile.enrolled_class.id
    classroom = get_object_or_404(Classroom, id=class_id)
    print(student)
    courses = Course.objects.filter(classroom_id=class_id)
    context = {'courses': courses,'classroom':classroom,'student':student,'s_profile':student_profile}
    return render(request,'all/student_home.html', context)

@login_required(login_url='my_login')
def s_todo(request):
    return render(request,'all/student_todo.html')

@login_required(login_url='my_login')
def t_home(request):
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    id_user= request.user.id
    classes = Classroom.objects.filter(teacher_user=id_user).order_by('level')
    num = len(classes)//4
    if (len(classes)%4 >0):
        num = num + 1
    context = {'classes':classes,'teacher':teacher,'t_profile':teacher_profile,'num':range(num)}
    return render(request,'all/teacher_home.html',context)

@login_required(login_url='my_login')
def t_class(request,id):
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    studentsP = StudentProfile.objects.filter(enrolled_class=id)
    problems = Problem.objects.filter(classroom_id=id)
    num_probs= len(problems)
    students = [profile.user for profile in studentsP]
    completed_problems = Done.objects.filter(problem__classroom_id=id, student__in=students).values('student').annotate(num_completed=Count('problem'))
    # Create a dictionary with student IDs as keys and the count of completed problems as values
    completed_problems_dict = {item['student']: item['num_completed'] for item in completed_problems}

     # Count the number of completed problems with result=True for each student
    completed_problems_true = Done.objects.filter(problem__classroom_id=id, student__in=students, result=True).values('student').annotate(num_completed=Count('problem'))

    # Create a dictionary with student IDs as keys and the count of completed problems with result=True as values
    completed_problems_true_dict = {item['student']: item['num_completed'] for item in completed_problems_true}

    # Prepare the data to be passed to the template
    student_data = []
    for student, studentP in zip_longest(students, studentsP):
        student_id = student.id
        num_completed = completed_problems_dict.get(student_id, 0)  # Get the count of completed problems or default to 0
        num_completed_true = completed_problems_true_dict.get(student_id, 0)  # Get the count of completed problems with result=True or default to 0
        student_data.append((student, studentP, num_completed, num_completed_true))
    print(student_data)
    context = {'students': student_data, 'num_probs': num_probs,'teacher':teacher,'t_profile':teacher_profile}
    return render(request, 'all/t_class.html', context)

@login_required(login_url='my_login')
def t_dashboard(request):
    return render(request,'all/teacher_dash.html')

@login_required(login_url='my_login')
def chat(request):
    #context = {'teachers': teachers}
    return render(request,'all/chat.html')

@login_required(login_url='my_login')
def t_courses(request,classroom_id):
    teacher_id = request.user
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    classroom_id=classroom_id
    courses = Course.objects.filter(teacher_id=teacher_id,classroom_id=classroom_id)
    classroom = get_object_or_404(Classroom, pk=classroom_id)
    context = {
        'classroom':classroom,
        'courses':courses,
        'classroom_id': classroom_id,
        'teacher_id':teacher_id,
        'teacher':teacher,'t_profile':teacher_profile
    }
    return render(request,'all/t_courses.html',context)

@login_required(login_url='my_login')
def t_problems(request,classroom_id):
    teacher_id = request.user
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    problems = Problem.objects.filter(teacher_id=teacher_id,classroom_id=classroom_id)
    classroom = get_object_or_404(Classroom, pk=classroom_id)
    context = {
        'classroom':classroom,
        'problems':problems,
        'classroom_id': classroom_id,
        'teacher_id':teacher_id,
        'teacher':teacher,'t_profile':teacher_profile
    }
    return render(request,'all/t_problems.html',context)
    
def t_createP(request,classroom_id):
    teacher_id = request.user
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    classroom = get_object_or_404(Classroom, pk=classroom_id)
    context = {
        'teacher':teacher,
        'classroom':classroom,
        'classroom_id': classroom_id,
        't_profile':teacher_profile,
        'teacher_id':teacher_id
    }
    return render(request, 'all/t_createP.html', context)


def t_createP2(request, classroom_id):
    teacher_id = request.user
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    classroom = get_object_or_404(Classroom, pk=classroom_id)
    
    if request.method == 'POST':
        text = request.POST.get('text')
        response = request.POST.get('response')
        classroom = Classroom.objects.get(id=classroom_id)
        problem = Problem.objects.create(
            text=text,
            answer=response,
            teacher=teacher,
            classroom=classroom)
    
    problems = Problem.objects.filter(teacher_id=teacher_id,classroom_id=classroom_id)

    context = {
        'problems':problems,
        'teacher': teacher,
        'classroom': classroom,
        't_profile':teacher_profile,
        'classroom_id': classroom_id,
        'teacher_id': teacher_id
    }
    return render(request, 'all/t_problems.html', context)
@login_required(login_url='my_login')
def helpView(request, problem_id):
    problem = get_object_or_404(Problem, id=problem_id)
    sentence1 = problem.text  # Utilisez la phrase provenant de l'objet Problem
    teacher = request.user
    teacher_profile=teacher.teacherprofile
    #supprimer la question
    # Trouver l'indice du premier point et du premier point d'interrogation
    index_dot = sentence1.rfind('.', 0, sentence1.find('?'))
    index_question = sentence1.find('?')

    # Vérifier si les indices des points et du point d'interrogation sont valides
    if index_dot != -1 and index_question != -1:
    # Supprimer la question de la phrase en conservant le texte avant le point et après le point d'interrogation
        sentence1 = sentence1[:index_dot+1] + sentence1[index_question:]

    # La fonction detecter_nombre_en_lettres, convert_number_to_digits, additionner_nombres et add_space_between_number_and_word
    # doivent être implémentées pour effectuer les transformations nécessaires sur la phrase. Assurez-vous de les inclure
    # et de les importer correctement ici.

    # Conversion de chaque nombre écrit en lettres dans la phrase en chiffres
    if detecter_nombre_en_lettres(sentence1):
        converted_sentence = convert_number_to_digits(sentence1)
        sentence_without_space = additionner_nombres(converted_sentence)
        sentence1 = add_space_between_number_and_word(sentence_without_space)

    nlp = spacy.load('en_core_web_lg')
    doc = nlp(sentence1)
    # Création du graphe de dépendances et dessin si au moins un nombre est présent
    has_num = False
    for token in doc:
        if token.like_num:
            has_num = True
            break
    
    if has_num:
        # Création du graphe de dépendances à partir de la phrase analysée avec Spacy
        edges = create_dependency_graph(doc)
        print(edges)
        # Dessin du graphe de dépendances à l'aide de NetworkX et Matplotlib
        draw_dependency_graph(edges)
        # Conversion du texte en discours en utilisant gTTS et enregistrement du résultat dans un fichier audio
        tts = gTTS(text=problem.text, lang='en')
        tts.save("mindLite/static/output.mp3")
        # Lecture du fichier audio en tant que bytes et encodage en base64 pour une utilisation dans la page HTML
        with open("mindLite/static/output.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Ajout du fichier audio TTS directement à la page HTML
        with open('mindLite/static/dependency_graph.html', 'r') as f:
            content = f.read()
        index = content.find('</body>')
        with open('mindLite/static/dependency_graph.html', 'w') as f:
            f.write(content[:index])
            f.write('<p align="center"><audio controls autoplay>\n')
            f.write('<source src="data:audio/mpeg;base64,{}" type="audio/mpeg">\n'.format(audio_base64))
            f.write('</audio>\n')
            f.write(content[index:])
            f.close()

    else:
        print("Le problème ne peut pas être présenté par un graphe.")

    # Rendre la vue avec les données nécessaires
    return render(request, 'all/t_aide.html', {
        'dependencies': edges,  # Passer les dépendances (edges) à la vue
        'audio_base64': audio_base64,
          'teacher': teacher,
        't_profile':teacher_profile  # Passer le fichier audio encodé en base64 à la vue
    })

@login_required(login_url='my_login')
def teacher_create_index(request,classroom_id):
    teacher = request.user
    teacher_id= teacher.id
    teacher_profile= teacher.teacherprofile
    context={
        'teacher':teacher,
        't_profile':teacher_profile,
        'teacher_id':teacher_id,
        'classroom_id':classroom_id
    }
    return render(request, 'all/t_index.html',context)
@login_required(login_url='my_login')
def teacher_courses(request,classroom_id):
    teacher = request.user
    teacher_id= teacher.id
    teacher_profile= teacher.teacherprofile
    context={
        'teacher':teacher,
        't_profile':teacher_profile,
        'teacher_id':teacher_id,
        'classroom_id':classroom_id
    }
    return render(request, 'all/t_index.html',context)



def create_course(request,classroom_id):


    if request.method == 'POST':
        title = request.POST.get('title')
        text = request.POST.get('text')
        image = request.FILES.get('image')
        teacher = request.user
        teacher_id= teacher.id
        teacher_profile= teacher.teacherprofile
        classroom = Classroom.objects.get(id=classroom_id)# Get the classroom instance
    
        summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summary_model = summary_model.to(device)

        def postprocesstext (content):
            final=""
            for sent in nltk.sent_tokenize(content):
                sent = sent.capitalize()
                final = final +" "+sent
            return final

        def summarizer(text,model,tokenizer):
            text = text.strip().replace("\n"," ")
            text = "summarize: "+text
            # print (text)
            max_len = 512
            encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            outs = model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            early_stopping=True,
                                            num_beams=3,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            min_length = 75,
                                            max_length=300)


            dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
            summary = dec[0]
            summary = postprocesstext(summary)
            summary= summary.strip()

            return summary
        
        def get_nouns_multipartite(content):
            out=[]
            
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=content,language='en')
            pos = {'PROPN','NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos)
            extractor.candidate_weighting(alpha=1.1,
                                            threshold=0.75,
                                            method='average')
            keyphrases = extractor.get_n_best(n=15)

            for val in keyphrases:
                out.append(val[0])

            return out
        
        def get_keywords(originaltext,summarytext):
            keywords = get_nouns_multipartite(originaltext)
            print ("keywords unsummarized: ",keywords)
            keyword_processor = KeywordProcessor()
            for keyword in keywords:
                keyword_processor.add_keyword(keyword)

            keywords_found = keyword_processor.extract_keywords(summarytext)
            keywords_found = list(set(keywords_found))
            print ("keywords_found in summarized: ",keywords_found)

            important_keywords =[]
            for keyword in keywords:
                if keyword in keywords_found:
                    important_keywords.append(keyword)

            return important_keywords[:4]
        
        
        summarized_text = summarizer(text,summary_model,summary_tokenizer)
        keyphrases = get_nouns_multipartite(text)
        imp_keywords = get_keywords(text,summarized_text)
        #return render(request,'user.html',{'name':keyphrases})
        # initialize the T5 model and tokenizer
        question_model = transformers.T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
        question_tokenizer = transformers.T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

        def get_question(context,answer,model,tokenizer):
            text = "context: {} answer: {}".format(context,answer)
            encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            outs = model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            early_stopping=True,
                                            num_beams=5,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            max_length=72)


            dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


            Question = dec[0].replace("question:","")
            Question= Question.strip()
            return Question
        question=[]
        for answer in imp_keywords:
            ques = get_question(summarized_text,answer,question_model,question_tokenizer)
            question.append(ques)
        # paraphrase-distilroberta-base-v1
        sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')# paraphrase-distilroberta-base-v1
        
        def get_distractors_wordnet(word):

            distractors=[]
            
            syn = wn.synsets(word,'n')[0]
                
            word= word.lower()
            orig_word = word
            if len(word.split())>0:
                word = word.replace(" ","_")
            hypernym = syn.hypernyms()
            if len(hypernym) == 0: 
                return distractors
            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name()
                #print ("name ",name, " word",orig_word)
                if name == orig_word:
                    continue
                name = name.replace("_"," ")
                name = " ".join(w.capitalize() for w in name.split())
                if name is not None and name not in distractors:
                    distractors.append(name)
        
            return distractors
            
        distractors_list=[]
        for key in imp_keywords:
            distractors_list.append(get_distractors_wordnet(key)[:4])

        course = Course.objects.create(
            title=title,
            text=text,
            summary=summarized_text,
            teacher=teacher,
            classroom=classroom,
            image=image,
        )
        for i in range(4):
            quiz = Quiz.objects.create(
                question=question[i],
                answer=imp_keywords[i],
                distractor1=distractors_list[i][0],
                distractor2=distractors_list[i][1],
                distractor3=distractors_list[i][2],
                course=course,
            )
        #distractors_list = get_distractors_wordnet("mouse")
        #return render(request, 'student_courses/details.html', {'course': course})
        return render(request, 'all/t_detail.html', {"summarized_text": summarized_text,
                            "keyphrases":keyphrases,
                            "imp_keywords":imp_keywords,
                            "question":question,
                            "distractors_list":distractors_list,
                            "teacher_id" : teacher_id,
                            "classroom_id" :classroom_id,
                            'course': course,'teacher':teacher,
        't_profile':teacher_profile})
@login_required(login_url='my_login')
def teacher_courses_details(request, course_id,classroom_id):
    teacher = request.user
    teacher_id=teacher.id
    teacher_profile= teacher.teacherprofile
    course = get_object_or_404(Course, pk=course_id)
    #quiz = get_object_or_404(Quiz, course_id=course_id)
    quiz = Quiz.objects.filter(course_id=course_id)
    
    return render(request, 'Teacher_courses/details_courses.html',{
                         'course': course,
                         'quiz':quiz,
                         'teacher_id':teacher_id,
                         'classroom_id':classroom_id,'teacher':teacher,
        't_profile':teacher_profile
                         })
@login_required(login_url='my_login')
def graph(request,classroom_id):
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    classroom = Classroom.objects.get(id=classroom_id)
    return render(request, 'all/t_editor.html',{'teacher':teacher,'t_profile':teacher_profile,'classroom':classroom})

@login_required(login_url='my_login')
def process_input(request,classroom_id):
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    classroom = Classroom.objects.get(id=classroom_id)
    if request.method == 'GET':
            input_text = request.GET.get('input_text', '')  # Retrieve the value of "input_text" parameter from the GET request
            # Perform further processing with the input_text variable
            text=input_text
            input_text = all.spacy_2.create_input_graph(input_text)
    else:
        input_text = ''
    return render(request, 'all/t_editor.html', {'input_text': input_text,'teacher':teacher,'t_profile':teacher_profile,'text':text,'classroom':classroom})



@login_required(login_url='my_login')
def t_chat(request,id):
    t_id = request.user.id
    teacher = request.user
    teacher_profile= teacher.teacherprofile
    parent = User.objects.get(id = id)
    p_profile = parent.parent_profiles
    chats = Chat.objects.filter(Q(sender=id) | Q(sender=t_id), Q(receiver=id) | Q(receiver=t_id))
    num = len(chats)
    users_with_chats = User.objects.filter(
        Q(chat_sender__receiver=teacher) | Q(chat_receiver__sender=teacher)
    ).exclude(id=teacher.id).distinct()
    context = {'chats': chats,'teacher':teacher,'t_profile':teacher_profile,'parent':parent,'p_profile':p_profile,'num':num,'users_with_chats': users_with_chats}
    return render(request,'all/t_chat.html',context)
@login_required(login_url='my_login')
def p_chat(request,id):
    p_id = request.user.id
    parent = request.user
    parent_profile= parent.parent_profiles
    teacher = User.objects.get(id = id)
    t_profile = teacher.teacherprofile
    chats = Chat.objects.filter(Q(sender=id) | Q(sender=p_id), Q(receiver=id) | Q(receiver=p_id))
    num = len(chats)
    users_with_chats = User.objects.filter(
        Q(chat_sender__receiver=parent) | Q(chat_receiver__sender=parent)
    ).exclude(id=teacher.id).distinct()
    context = {'chats': chats,'teacher':teacher,'t_profile':t_profile,'parent':parent,'p_profile':parent_profile,'num':num,'users_with_chats': users_with_chats}
    return render(request,'all/p_chat.html',context)

@login_required(login_url='my_login')
def my_logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url='my_login')
def save_message(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        receiver_id = request.POST.get('receiver_id')
        print(receiver_id)
        if message and receiver_id:
            sender = request.user
            receiver = User.objects.get(id=receiver_id)
            
            # Create a new Chat object and savwe the message
            chat = Chat(sender=sender, receiver=receiver, content=message)
            chat.save()
            return redirect('t_chat', id=receiver_id)
        
    return HttpResponseBadRequest()  

@login_required(login_url='my_login')
def save_message_p(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        receiver_id = request.POST.get('receiver_id')
        if message and receiver_id:
            sender = request.user
            receiver = User.objects.get(id=receiver_id)
            
            # Create a new Chat object and savwe the message
            chat = Chat(sender=sender, receiver=receiver, content=message)
            chat.save()
            print(receiver.id)
            return redirect('p_chat', id=receiver.id)
        
    return HttpResponseBadRequest()  


@login_required(login_url='my_login')
def upload_photo(request):
    if request.method == 'POST':
        # Retrieve the uploaded photo file from request.FILES
        photo_file = request.FILES.get('photo')
        if photo_file:
            # Update the user's photo attribute with the uploaded file
            user = request.user
            user.photo = photo_file
            user.save()
            print(user.photo)
            
    return redirect('t_home')
@login_required(login_url='my_login')
def upload_photo_p(request):
    if request.method == 'POST':
        # Retrieve the uploaded photo file from request.FILES
        photo_file = request.FILES.get('photo')
        if photo_file:
           
            user = request.user
            user.photo = photo_file
            user.save()
            print(user.photo)
            
    return redirect('p_home')
@login_required(login_url='my_login')
def upload_photo_s(request):
    if request.method == 'POST':
        # Retrieve the uploaded photo file from request.FILES
        photo_file = request.FILES.get('photo')
        choice = request.POST.get('choice')
        
        if photo_file:
            # Update the user's photo attribute with the uploaded file
            user = request.user
            user.photo = photo_file
            user.save()
            print(user.photo)
            
    return redirect('s_home')

@login_required(login_url='my_login')
def upload_photo_S(request):
    if request.method == 'POST':
        # Retrieve the uploaded photo file from request.FILES
        photo_file = request.FILES.get('photo')
        choice = request.POST.get('choice')
        device = "cuda"
        model_id_or_path = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        prompt_superhero = "Turn this kid into a superman"
        prompt_warrior="A photorealistic image of a fierce men viking warrior in the midst of battle."
        prompt_princess="gorgeous disney princess jack black, professionally retouched, muted colors, soft lighting, realistic, smooth face, full body shot, torso, dress, perfect eyes, sharp focus on eyes, 8 k, high definition, insanely detailed, intricate, elegant, art by j scott campbell and artgerm "
        images = pipe(prompt=prompt_princess , image=init_image, strength=0.75, guidance_scale=7.5).images 
        if photo_file:
            # Update the user's photo attribute with the uploaded file
             # Update the user's photo attribute with the uploaded file
            init_image = Image.open(photo_file).convert("RGB")
            init_image = init_image.resize((768, 512))
            images = pipe(prompt=prompt_princess , image=init_image, strength=0.75, guidance_scale=7.5).images
            user = request.user
            user.photo = photo_file
            user.save()
            print(user.photo)
            
    return redirect('p_home')
