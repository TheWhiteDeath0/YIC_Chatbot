from django.contrib import admin
from .models import Intent, Pattern, Response

class PatternInline(admin.TabularInline):
    model = Pattern
    extra = 1

class ResponseInline(admin.TabularInline):
    model = Response
    extra = 1

class IntentAdmin(admin.ModelAdmin):
    list_display = ('tag', 'description')
    inlines = [PatternInline, ResponseInline]

admin.site.register(Intent, IntentAdmin)
