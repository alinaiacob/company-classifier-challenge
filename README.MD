Prin intermediul acestui proiect am experimentat mai multe strategii privind clasificare companiilor in functie de label-urile disponibile
1. Prima abordare a fost utilizarea unui model mult mai rapid si pe baza tuturor informatiilor despre companie sa utilizez embeddings
2. A doua abordare a fost legata si de o clasificare a label-urilor din taxonomy -> si sa se adauge pentru fiecare label sector, category. In acest fel am exclus o parte din companiile care nu se incadrau in sector, category atribuit label-ului. Ulterior am folosit o analiza de cluster(hierarchical)- dupa am folosit embeddings si mapare pe datele companiei
3. A treia abordare a fost utilizarea unui model mult mai robust si care consuma mult mai multe resurse - iar pentru date care sunt foarte complexe - trebuie luat in vedere acest aspect

Am inclus rezultatele atat pentru a ii atribui unei companii doar un singur label -> rezultatele se gasesc la calea:
results/one_label/companies_predicted_hierarchical_optimized.csv
results/one_label/companies_predicted_labels_only.csv
results/one_label/companies_predicted_labels_only_larger_model.csv


Am inclus rezultatele atat pentru a ii atribui unei companii doua label-uri -> rezultatele se gasesc la calea:
results/more_labels/companies_predicted_hierarchical_optimized.csv
results/more_labels/companies_predicted_labels_only.csv
results/more_labels/companies_predicted_labels_only_larger_model.csv


Sursele initiale de date se regasesc in folderul datasets. In acest folder  in fisierul este prezenta noua coloana ml_insurance_challenge.csv cu label-urile aferente din taxonommy


Modele utilizate se regasesc la calea:
models/flat-model.py
models/hierarchical-model.py

In fisierul add_results.py am adaugat in fisierul initiat label-ul aferent clasificarii (am folosit modelul Mpnet)

De asemenea, am realizat si un website de prezentare cu Streamlit, in care se poate interactiona cu rezultatele, si fisierul are numele results_insights.py
In acest fel am comparat rezultatele obtinute si am putut vedea o imbunatatire a rezultatelor atunci cand s-a utilizat un model mai robust si complex.

Din aceasta analiza, am ajuns la concluzia ca cea mai buna abordare este cea in care se utilizeaza un model mult mai complex, cu precizarea ca si consumul de resurse va creste semnificativ


