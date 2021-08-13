################# KÜTÜPHANELERİN YÜKLENMESİ ##############
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


pd.set_option("display.max_columns",None) # Maximum sütun sayısını göster.

df=pd.read_csv("train_fraud.csv")  # Veri setinin yüklenmesi.

df.head(5)  # İlk beş satırın gösterilmesi.

df.isnull().sum().sum() # Veride boş değer var mı kontrolü
df=df.drop(["Id","nameOrig","nameDest"],axis=1)  # Eğitimde işimize yaramayacak sütunları
                                                 # atar.

cat_cols=["action"]  # one_hot encoderı'a kategorik verileri vermek için oluşturduğumuz
                     # cat_cols listesi.

df["action"].value_counts().nunique()

df=pd.get_dummies(df,columns=cat_cols) # one_hot_encoder uygulaması

df.head(5) #İlk beş satırın gösterilmesi.



x=df.drop("isFraud",axis=1)  # Veri eğitilirken kullanacağımız bağımsız değişkenler.
y=df["isFraud"]   # Veri eğitilirken kullanacağımız bağımlı değişken.


# Verinin train ve test olarak 0.33 şeklinde ayrılması.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=15)




y_train.value_counts()  # Imbalanced veri kontrolü


# over_sampling metodlarından SMOTE olanı kullanarak imbalanced verinin üstesinden geldik.
from imblearn.over_sampling import SMOTE
sample= SMOTE()
x_smote, y_smote = sample.fit_resample(x_train,y_train)

y_smote.value_counts() # Üstesinden gelinip veya gelinmediğinin kontrolü.

df.head(5)
# Bağımsız değişkenlerin değerlerini belirli bir aralığa sıkıştırmak.
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
x_smote = ms.fit_transform(x_smote)
x_test = ms.transform(x_test)

###############Model Seçimi######################

# İlk modelimiz olan GradientBoostingClassifier modelinin kurulumu.
gbm=GradientBoostingClassifier(random_state=14)
gbm.fit(x_smote,y_smote)
y_pred_gbm=gbm.predict(x_test)


# İlk modelimizin confusion_matrixinin oluşturulması
from sklearn.metrics import confusion_matrix
cm_gbm=confusion_matrix(y_test,y_pred_gbm)
print(cm_gbm)

# Sonucumuz = [[781070   1888]
#            [     3     65]]

# İlk modelimizin Recall değerini öğrenmek için classification_report metodunun oluşturulması
from sklearn.metrics import classification_report
report_gbm=classification_report(y_test,y_pred_gbm)
print(report_gbm)

# Sonucumuz =  precision    recall  f1-score   support
#          0       1.00      1.00      1.00    782958
#          1       0.03      0.96      0.06        68






# İkinci modelimiz olan AdaBoostClassifier modelinin kurulumu.
abc=AdaBoostClassifier(random_state=14)
abc.fit(x_smote,y_smote)
y_pred_abc= abc.predict(x_test)


# İkinci modelimizin confusion_matrixinin oluşturulması
from sklearn.metrics import confusion_matrix
cm_abc=confusion_matrix(y_test,y_pred_abc)
print(cm_abc)

# Sonucumuz =  [[780054   2904]
#              [     3     65]]



# İkinci modelimizin Recall değerini öğrenmek için classification_report metodunun oluşturulması
from sklearn.metrics import classification_report
report_abc=classification_report(y_test,y_pred_abc)
print(report_abc)

#Sonucumuz =  precision    recall  f1-score   support
#          0       1.00      1.00      1.00    782958
#          1       0.02      0.96      0.04        68

# Birçok modeli denedikten sonra aldığımız en iyi iki sonuçlu modeller bunlardır.
