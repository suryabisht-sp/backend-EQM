from django.db import models

# Create your models here.

class appnew(models.Model):
    title=models.CharField(max_length=50)
    expenses=models.CharField(max_length=6)

    def __str__(self) -> str:
        return f"{self.title} {self.expenses}"
    
    from django.db import models

class Expense(models.Model):
    date = models.DateField()
    category = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    def __str__(self):
        return f"Date: {self.date} - Category: {self.category} - Amount: ${self.amount}"
