from . import views
from django.urls import path


urlpatterns = [
    path('analyze_first/', views.analyze_first, name='analyze_first'),
    path('analyze_sentiment/', views.analyze_sentiment, name='analyze_sentiment'),
]
