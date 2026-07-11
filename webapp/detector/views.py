from django.shortcuts import render

from . import ml


def index(request):
    """Single view: GET shows the empty form, POST runs a prediction and
    re-renders the same page with results. No models/database involved --
    this demo is stateless by design (see project README for why)."""
    context = {}

    if request.method == "POST":
        email_text = request.POST.get("email_text", "").strip()
        if email_text:
            context["result"] = ml.predict(email_text)
            context["email_text"] = email_text
        else:
            context["error"] = "Please paste some email text to classify."

    return render(request, "detector/index.html", context)
