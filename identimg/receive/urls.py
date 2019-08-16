from django.conf.urls import url
from . import views
app_name = 'receive'

urlpatterns = [
    url(r'^ident_img/$', views.ident_img, name='ident_img')
]