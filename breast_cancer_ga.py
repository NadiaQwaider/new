# ========================================================
# تطبيق عملي للخوارزمية الجينية لاختيار الخصائص (Feature Selection)
# على داتاسيت سرطان الثدي (Breast Cancer dataset)
# ========================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import random
import arabic_reshaper
from bidi.algorithm import get_display

# 1. تحميل البيانات
data = load_breast_cancer()
X = data.data
y = data.target
n_features = X.shape[1]
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 2. تقسيم البيانات (تدريب + اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------
# إعداد الخوارزمية الجينية GA
# ----------------------------

# دالة لياقة (Fitness Function): تقيس الدقة باستخدام الخصائص المختارة
def fitness(individual):
    # تحويل الفرد (سلسلة من 0/1) إلى قائمة خصائص
    selected_features = [i for i in range(n_features) if individual[i] == 1]
    if len(selected_features) == 0:  # لو ما اختار ولا خاصية
        return 0,
    
    # تدريب نموذج بسيط (Logistic Regression) على الخصائص المختارة
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train[:, selected_features], y_train)
    y_pred = clf.predict(X_test[:, selected_features])
    acc = accuracy_score(y_test, y_pred)
    return acc,

# إعداد مكتبة DEAP (الخاصة بالـ GA)
from deap import base, creator, tools

# إنشاء نوع جديد: "كل فرد = قائمة من 0/1" بطول عدد الخصائص
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)  # جينات ثنائية (0/1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# تسجيل العمليات
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)     # تقاطع (Crossover)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # طفرة (Mutation)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------------
# تنفيذ الخوارزمية الجينية
# ----------------------------
def run_ga():
    pop = toolbox.population(n=30)  # حجم المجتمع
    NGEN = 20   # عدد الأجيال
    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # تقاطع + طفرة
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # تقييم الأفراد
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring

        # أفضل فرد في هذا الجيل
        fits = [ind.fitness.values[0] for ind in pop]
        text1= "الجيل"
        reshaped_text1 = arabic_reshaper.reshape(text1)    # correct its shape
        bidi_text1 = get_display(reshaped_text1)   # correct its direction

        text2= "أفضل دقة "
        reshaped_text2 = arabic_reshaper.reshape(text2)    # correct its shape
        bidi_text2 = get_display(reshaped_text2)           # correct its direction

        text3= " أفضل مجموعة خصائص مختارة: "
        reshaped_text3 = arabic_reshaper.reshape(text3)    # correct its shape
        bidi_text3 = get_display(reshaped_text3)  # correct its direction

        text4= " دقة النموذج النهائي  "
        reshaped_text4 = arabic_reshaper.reshape(text4)    # correct its shape
        bidi_text4 = get_display(reshaped_text4)  # correct its direction
       
        print(f"{bidi_text1} {gen+1}: {bidi_text2} = {max(fits):.4f}")

    best_ind = tools.selBest(pop, 1)[0]
    print(f"\n✅ {bidi_text3}")
    print([i for i in range(n_features) if best_ind[i] == 1])
    print(f"{bidi_text4} = {fitness(best_ind)[0]:.4f}")

# تشغيل الخوارزمية
run_ga()
