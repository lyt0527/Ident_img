from django.http import JsonResponse
from receive.IDCardRecognise import IDRecognise


def ident_img(request):
    path = request.POST.get("url")
    t = IDRecognise()
    number = t.getCardID(path)
    # print(type(number))
    return JsonResponse({"msg":'ok', "data":number})













