from . import views
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',  views.index, name='index'),
    path('term/',  views.autocomplete, name='autocomplete'),
    path('path/<str:src>/<str:gol>', views.search_dij, name='search_dij'),
    path('path/<str:source>/', views.dynamic_url, name='dynamic_url'),
    path('generate-home/', views.gen_view, name='gen'),
    path('generate-qr/<str:location>/', views.generate_qr, name='generate-qr'),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

handler = 'navApp.views.error'