from django.contrib import admin
from myapp.models import appnew
from .models import Expense

admin.site.register(appnew)

admin.site.register(Expense)

# Register your models here.
