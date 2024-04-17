from django import forms

class MyForm(forms.Form):
    my_input = forms.CharField(label='Enter Text', max_length=100)