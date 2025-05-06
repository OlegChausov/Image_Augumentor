This script creates **new versions of images**, primarily designed for small-sized datasets due to computing limitations. It detects **patterns in the original image**, modifies part of it, and then saves the transformed version.  

## ⚙️ How It Works  
Built on **Scikit-Learn regression models**, allowing users to choose from three different kernels:  

- **📏 Linear Kernel** – Basic model. Some patterns may be lost if the image lacks **clear splitting lines**.  
- **🌀 Polynomial Kernel** – More advanced, suitable for **detailed images**.  
- **🔥 RBF Kernel (Radial Basis Function)** – The most powerful option, great for **regression tasks**, but beware of **high CPU load**.  

Additionally, users can adjust the **Test Model size** in the range `0-1`, defining how much of the image should be modified—**larger values require more processing power**.  

✅ **Intel CPUs** can use **SK-Learn Intel Patch** for optimization.  

## 🛠️ Technologies  
- 🐍 **Python**  
- 🤖 **Scikit-Learn** ([GitHub](https://github.com/scikit-learn/scikit-learn))  
- 🏛️ **Pandas**  


Этот скрипт создает **новые версии изображений**, предназначенные в основном для небольших размеров (из-за вычислительных ограничений). Он анализирует **паттерны исходного изображения**, изменяет его часть и затем сохраняет трансформированную версию.  

## ⚙️ Как работает  
Основан на **регрессионных моделях** `Scikit-Learn`. Пользователь может выбрать один из трех вариантов ядер:  

- **📏 Линейное ядро (Linear Kernel)** – базовая модель. Некоторые паттерны могут теряться, если изображение **не имеет четких разделительных линий**.  
- **🌀 Полиномиальное ядро (Polynomial Kernel)** – более сложная модель, подходит для **детализированных изображений**.  
- **🔥 RBF ядро (Radial Basis Function Kernel)** – самое мощное, отлично подходит для **задач регрессии**, но может сильно нагружать **CPU**.  

Также можно настроить размер **Test Model** в диапазоне `0-1`, определяя, **какая часть изображения будет изменена** — **чем больше, тем выше требования к вычислительным ресурсам**!  

✅ **Intel CPU** могут использовать **SK-Learn Intel Patch** для оптимизации вычислений.  

## 🛠️ Используемые технологии  
- 🐍 **Python**  
- 🤖 **Scikit-Learn** ([GitHub](https://github.com/scikit-learn/scikit-learn))  
- 🏛️ **Pandas**  


![Screeshots of app](/assets/images/saaampl.png)
![Screeshots of app](/assets/images/saaampl_result_Poly_test_size_.png)
![Screeshots of app](/assets/images/saaampl_result_RBF_test_size_0.9.png)
