from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from .test import output


# Create your views here.

def index(request):
    return render(request, 'LIVE/index.html', {})


def gen(camera):
    while True:
        global i
        i = 0
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def live_feed(request):
    return StreamingHttpResponse(gen(output()), content_type='multipart/x-mixed-replace; boundary=frame')


def test(request):
    return "Hi baba"
