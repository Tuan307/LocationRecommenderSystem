from TravelRecommenderApp.views import recommendation
from django.urls import path

urlpatterns = [
   path('travel/',recommendation)
]