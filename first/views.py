# views.py
from django.shortcuts import render
from .sentiment_analysis import predict_sentiment
from django.http import HttpResponse
def analyze_first(request):
    return render(request, 'sentiment_form.html')

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text')  # Get text input from a form

        # Perform sentiment analysis using the predict_sentiment function
        predicted_sentiment = predict_sentiment(text)
        # Return the result to the user
        return render(request, 'sentiment_result.html', {'text': text, 'predicted_sentiment': predicted_sentiment})
    
    
    
