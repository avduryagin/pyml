1. Установить [python-3.8.7](https://www.python.org/downloads/release/python-387/)
2. Распаковать архив. Следующие шаги относительно директории c распакованным приложением
3. Создать виртуальную среду командой
   py -3.8 -m venv .venv
4. Активировать виртуальную среду командой
   .\\.venv\Scripts\activate
5. Установить пакеты:  
   Скачиваем у себя необходимые для работы пакеты в папку packages  
   pip download -r requirements.txt -d packages  
   После папку с пакетами копируем на сервер и устанавливаем пакеты  
   pip install -r requirements.txt --no-index --find-links file://D:\ois\iis-py-app\packages
6. Настроить IIS
