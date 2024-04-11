from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Lines, Points, mapImage,graph
from .views import creat_map
# Register your models here.

class PointsAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    ...
class graphAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    ...
class LinesAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    ...
class ImgAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    ...
admin.site.register(Lines, LinesAdmin)
admin.site.register(Points, PointsAdmin)
admin.site.register(mapImage, ImgAdmin)

def Update_Graph(self, request, queryset):
    # Create a temporary variable to store the attributes
    selected_objects = self.get_queryset(request).filter(pk__in=request.POST.getlist('_selected_action'))
    # Loop through the selected objects in the queryset
    for obj in selected_objects:
        print(obj.floor)
        # Call the function that processes the attributes
        creat_map(obj.floor)  # Replace 'your_function' with your custom function

    # Display a success message to the user
    self.message_user(request, "Update performed successfully.")

Update_Graph.short_description = "Update Graph"
@admin.register(graph)
class adminCreateGraph(admin.ModelAdmin):
    actions = [Update_Graph]
