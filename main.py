
#  +=========================================================================================+
# || Stomes platformasi uchun maxsus yuzni tanish dasturi                                   ||
# || ushbu dasturga foydalanuvchi rasmi yuboriladi va dastur foydalanuvchini tanib beradi   ||
# || agar foydalanuvchi bazada bo'lmasa bazada mavjud emasligini chiqaradi                  ||
# || agar yangi foydalanuvchi kelsa uni bazaga kiritadi                                     ||
#  +=========================================================================================+

print("""
 ########::::'###:::::'######::'########::'######::'########::'#######::'##::::'##:'########:
 ##.....::::'## ##:::'##... ##: ##.....::'##... ##:... ##..::'##.... ##: ###::'###: ##.....::
 ##::::::::'##:. ##:: ##:::..:: ##::::::: ##:::..::::: ##:::: ##:::: ##: ####'####: ##:::::::
 ######:::'##:::. ##: ##::::::: ######:::. ######::::: ##:::: ##:::: ##: ## ### ##: ######:::
 ##...:::: #########: ##::::::: ##...:::::..... ##:::: ##:::: ##:::: ##: ##. #: ##: ##...::::
 ##::::::: ##.... ##: ##::: ##: ##:::::::'##::: ##:::: ##:::: ##:::: ##: ##:.:: ##: ##:::::::
 ##::::::: ##:::: ##:. ######:: ########:. ######::::: ##::::. #######:: ##:::: ##: ########:
..::::::::..:::::..:::......:::........:::......::::::..::::::.......:::..:::::..::........::
""")


import os
import uvicorn
from fastapi import FastAPI, UploadFile, Request, File, Form
from fastapi.responses import FileResponse

from deepface import DeepFace

# Create an instance of the FastAPI class
app = FastAPI()

# Ma'lumotlarni yuklash uchun
DB_DIR = './base/'
# Solishtiriladigan rasimlarni saqlash uchun asosiy katalog
UPLOAD_DIRECTORY = './uploads/'
MODEL_NAME = 'Facenet'

# foydalanuvchini bazadan tekshirish
@app.post("/identify")
async def identify(img: UploadFile = File(...)):
    """foydalanuvchini rasmi yuboriladi va dastur 
       foydalanuvchini bazada bor yoki yo'qligini tanib beradi

    Args:
        img: file
    
    Returns:
        foydalanuvchini bazada bor yoki yo'qligini chiqaradi
        json: {
            'is-user': bool
        }
    """

    files_count = len(os.listdir(UPLOAD_DIRECTORY))    
    file_name = str(files_count) + " " + img.filename
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await img.read())

    results = DeepFace.find(
        img_path=file_path,
        db_path=DB_DIR
        )
        
    # vaqtinchalik faylni olib tashlandi
    
    # bazada topilgan rasimlar
    images = [row["identity"] for index, row in results[0].iterrows()]

    results = []
    for i in images:
        img1 = DeepFace.detectFace(i)
        img2 = DeepFace.detectFace(file_path)

        result = DeepFace.verify(img1_path=i, img2_path=file_path, model_name=MODEL_NAME)
        if result["verified"]:
            results.append(file_path)
    
    os.remove(file_path)

    # javobni yuborish
    return {
        'is-user': bool(results),
        'images': results
    }

# foydalanuvchi yaratish uchun
@app.post("/create")
async def create_user(img: UploadFile = File(...), user_id: str = Form(...)):
    """foydalanuvchi yaratiladi va uni bazaga kiritiladi

    Args:
        img: file
        user_id: str
    
    Returns:
        bazaga kiritiladi
        json: {
            'status': bool,
            'message': _message_
        }
    """
    
    # foydalanuvchi folderi uchun yo'q
    folder_path = DB_DIR + user_id
    # foydalanuvchi uchun folder yaratildi
    os.makedirs(folder_path, exist_ok=True)

    # fayil nomi
    file_name = img.filename
    # fayil yo'li
    file_path = folder_path + "/" + file_name

    # user bazaga kiritildi
    with open(file_path, "wb") as buffer:
        buffer.write(await img.read())
    
    # user yaratildi
    return {
        'status': True,
        'message': 'User created'
    }

# userni rasmini olish
@app.get("/images")
async def get_photo(request: Request):
    """path bo'yicha bazadan userni rasimlarini olish

    Args:
        request (Request): path = image file path

    Returns:
        img: binary file
    """

    # query parametrlar
    query_params = request.query_params
    # fayl manzili
    file_path = query_params.get('path')

    # fayl topilmasa javob qaytarish
    if not file_path:
        return {
            'status': False,
            'message': 'file topilmadi'
        }
    
    # faylni yuborish 
    return FileResponse(path=file_path)

# RUN
if __name__ == "__main__":
    uvicorn.run('main:app', host="127.0.0.1", port=8000, reload=True)