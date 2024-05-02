import heapq
import io
import math
import os
from django.db.models import Q
import qrcode
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.files.temp import NamedTemporaryFile
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .models import mapImage, Points, Lines
from PIL import Image, ImageDraw, ImageFont
import json
from .models import graph
from django.contrib import messages
from django.core.exceptions import ObjectDoesNotExist


class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

def autocomplete(request):
    if 'term' in request.GET:
        qs = Points.objects.filter(Q(title__icontains=request.GET.get('term'))| Q(alt__icontains=request.GET.get('term')))[:5] # Limit to 10 results
        alts = qs.values_list('alt', flat=True) # Directly extract titles
        all_names= []
        for string in alts:

            # Split the string using the comma delimiter
            words = string.split(",")
            # Use list comprehension to remove leading/trailing whitespaces (optional)
            words = [word.strip() for word in words]
            if len(words)>1:
                words = [substring(words,request.GET.get('term'))]
            # Add the extracted words from this string to the all_words list
            all_names.extend(words)
        unique_list = set(all_names)
        all_names = unique_list
        all_names = [name for name in all_names if name != "null"]
        return JsonResponse(list(all_names), safe=False)  # Convert queryset to list for JSON response
    return render(request, "base.html")
def substring(string_array, substring):
  """
  This function takes an array of strings and a substring as input.
  It iterates through the array and returns the first string that contains the substring.
  If no string contains the substring, it returns None.
  """
  for string in string_array:
    if substring in string:
      return string
  return None
'''
# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minDistance(vertices, dist, sptSet):
    # Initialize minimum distance for next node
    min = 1e7
    min_index = 0
    # Search not nearest vertex not in the
    # shortest path tree
    for v in range(vertices):
        if dist[v] < min and sptSet[v] == False:
            min = dist[v]
            min_index = v

    return min_index


# Function that implements Dijkstra's single source
# shortest path algorithm for a graph represented
# using adjacency matrix representation
def dijkstra(Gph, src, goal):
    deGraph = json.loads(Gph.graph)
    dist = [1e7] * Gph.vertices
    dist[src.id%1000] = 0
    sptSet = [False] * Gph.vertices
    parent = [-1] * Gph.vertices  # Track the parent of each vertex in the shortest path

    for cout in range(Gph.vertices):
        u = minDistance(Gph.vertices, dist, sptSet)
        sptSet[u] = True
        for v in range(Gph.vertices):
            if (deGraph[u][v] > 0 and
                    sptSet[v] == False and
                    dist[v] > dist[u] + deGraph[u][v]):
                dist[v] = dist[u] + deGraph[u][v]
                parent[v] = u  # Set the parent of v as u

    return getSolution(dist, parent, src, goal)


def getSolution(dist, parent, src, goal):
    print("Vertex\tDistance\tPath")
    print(goal.title)
    for i in range(len(dist)):
        if i == goal.id:
            path = getPath(parent, src)
            print(f"{i}\t\t{dist[i]}\t\t{path}")
            return path


def getPath(parent, v):
    path = []
    while v != -1:
        path.append(v)
        v = parent[v]
    path.reverse()
    return path
'''
def gen_view(request):
    return render(request, 'geterate.html')

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height
def generate_qr(request, location):
    # Set up QR code generator
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(location)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    # Prepare to draw the text
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Load default font
    text = location
    text_width, text_height = textsize(text, font=font)

    # Calculate text positioning (centered)
    img_width, img_height = img.size
    x = (img_width - text_width) / 2
    y = img_height  # Position at the bottom of the QR code
    draw.text((x, y), text, font=font, fill="black")

    # Finalize and return the image response
    response = HttpResponse(content_type="image/png")
    img.save(response, "PNG")
    return response
def dijkstra(graph, src, goal):
    n = graph.vertices+1
    deGraph = json.loads(graph.graph)
    distances = [float('inf')] * n
    distances[src] = 0
    queue = [(0, src)]
    print(queue)
    while queue:
        current_dist, current_node = heapq.heappop(queue)
        # If the current node is the goal node, we have found the shortest path
        if current_node == goal:
            break

        # Skip this iteration if the current distance is greater than the stored distance
        if current_dist > distances[current_node]:
            continue

        for neighbor in range(n):
            weight = deGraph[current_node][neighbor]
            if weight > 0:
                distance = current_dist + weight

                # If a shorter path to the neighbor is found, update the distance
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))
    if distances[goal] == float('inf'):
        return None  # No path found

    # Reconstruct the path from the goal node to the source node
    path = []
    current = goal
    while current != src:
        path.append(current)
        for neighbor in range(n):
            if deGraph[current][neighbor] > 0 and distances[current] == distances[neighbor] + deGraph[current][neighbor]:
                current = neighbor
                break
    path.append(src)
    path.reverse()

    return path
# Create your views here.
def index(request):
    if request.method == 'POST':
        source = request.POST.get('location')
        goal = request.POST.get('destination')
        if source == '' or goal == '':
            messages.error(request,
                           'فضلا أدخل وجهتك او موقعك بشكل صحيح.')
            GroundFlr = get_object_or_404(mapImage, floor=0)
            Ground_url = GroundFlr.path.url
            firstFlr = get_object_or_404(mapImage, floor=1)
            first_url = firstFlr.path.url
            secFlr = get_object_or_404(mapImage, floor=2)
            sec_url = secFlr.path.url
            context = {
                'GFloor_url': Ground_url,
                '1stFloor_url': first_url,
                '2ndFloor_url': sec_url,
            }
            return render(request, "base.html", context)
        return redirect(reverse('search_dij', kwargs={'src': source, 'gol': goal}))
    GroundFlr = get_object_or_404(mapImage, floor=0)
    Ground_url = GroundFlr.path.url
    firstFlr = get_object_or_404(mapImage, floor=1)
    first_url = firstFlr.path.url
    secFlr = get_object_or_404(mapImage, floor=2)
    sec_url = secFlr.path.url
    context = {
        'GFloor_url': Ground_url,
        '1stFloor_url': first_url,
        '2ndFloor_url': sec_url,
    }
    return render(request, "base.html", context)
def dynamic_url(request, source):
    if request.method == 'POST':
        new_user_input = request.POST.get('location')
        return redirect(reverse('dynamic_url', kwargs={'source': new_user_input}))
    GroundFlr = get_object_or_404(mapImage, floor=0)
    Ground_url = GroundFlr.path.url
    firstFlr = get_object_or_404(mapImage, floor=1)
    first_url = firstFlr.path.url
    secFlr = get_object_or_404(mapImage, floor=2)
    sec_url = secFlr.path.url
    context = {
        'GFloor_url': Ground_url,
        '1stFloor_url': first_url,
        '2ndFloor_url': sec_url,
        'source': source
    }
    return render(request, 'Base.html', context)
def search_dij(request, src, gol):
    #if request.method == 'POST':
    default_imgs = mapImage.objects.filter(title__in=["1stFloor", "2ndFloor", "GFloor"])
    context = {img.title + "_url": img.path.url for img in default_imgs}
    buttons ={"select"+str(img.floor): "" for img in default_imgs}
    #location = request.POST.get('location')
    location = src
    #destination = request.POST.get('destination')
    destination = gol
    if location == '' or destination == '':
        messages.error(request,
                       'فضلا أدخل وجهتك او موقعك بشكل صحيح.')
        return render(request, "base.html", context)
    source = Points.objects.filter(alt__icontains=location)
    print("source: ",source)
    if len(source)>1:
        try:
            source = Points.objects.get(alt=location)
        except:
            messages.error(request, 'لايمكن اختيار هذه النقطة كنقطة بداية, فضلا قم بتغيير النقطة أو قم بمسح اقرب QR code')
            return render(request, "base.html", context)
    elif(len(source) == 1):
        source = source.first()
        print("source: ", source)
    else:
        messages.error(request,
                       'لم يتم العثور على النقطة.')
        return render(request, "base.html", context)
    goal = Points.objects.filter(alt__icontains=destination, floor=source.floor)
    if len(goal) == 0:
        goal = Points.objects.filter(alt__icontains=destination).first()
        print("goal(1):", goal)
        if goal == None:
            messages.error(request, 'Destination Not Found.')
            return render(request, "base.html", context)
    elif len(goal)>1:
        goal = nearest_point(goal,source)
    else:
        print("goal =",goal)
        goal = goal.first()
    UoD = source.floor - goal.floor
    floors_involved = [source.floor, goal.floor]
    if abs(UoD) >= 2:
        floors_involved.append(3-(source.floor+goal.floor))
    mapsImgs =[]
    selected =[]
    if source.floor == goal.floor:
        IDfactor = (source.floor * 1000) if source.floor != 0 else 9999
        grphObj = graph.objects.get(floor=source.floor)
        path = dijkstra(grphObj, source.id % IDfactor, goal.id % IDfactor)
        print(path)
        mapObj = mapImage.objects.get(floor=source.floor)
        img = mapObj.path
        new_img = draw_path(img, path, source.floor)  # Assuming this returns a modified image
        mapsImgs.append(mapObj.title)
        selected.append(mapObj.floor)
        new_img_io = io.BytesIO()
        new_img.save(new_img_io, format='JPEG')
        new_img_content = ContentFile(new_img_io.getvalue())
        floorImg = mapImage.objects.get(title=mapObj.title + "_path")
        existing_file_path = floorImg.path.path
        if os.path.exists(existing_file_path):
            os.remove(existing_file_path)
        file_name = os.path.basename(existing_file_path)
        floorImg.path.save(file_name, new_img_content, save=True)
    else:
        for floor in floors_involved:
            print(floor)
            IDfactor = (floor * 1000) if floor != 0 else 9999
            grphObj = graph.objects.get(floor=floor)
            if(UoD < 0):
                points = Points.objects.filter(Q(title__icontains='stair') | Q(title__icontains='upstair'), floor=floor)
            elif(UoD > 0):
                points = Points.objects.filter(Q(title__icontains= 'stair')| Q(title__icontains='downstair'), floor=floor)
            srcStair = nearest_point(points,source)
            goalStair = Points.objects.filter(floor=goal.floor, pointX=srcStair.pointX, pointY=srcStair.pointY).first()
            if floor == source.floor:
                print(source.id % IDfactor)
                # Logic for source floor to nearest stair
                path = dijkstra(grphObj, source.id % IDfactor, srcStair.id % IDfactor)
                print(path)
            elif floor == goal.floor:
                # Assuming similar logic for goal floor from stair to goal
                path = dijkstra(grphObj, goalStair.id % IDfactor, goal.id % IDfactor)
            else:
                # Handle intermediary floors if any, assuming direct stair to stair
                continue  # Adjust as needed

            mapObj = mapImage.objects.get(floor=floor)
            img = mapObj.path
            new_img = draw_path(img, path, floor)  # Assuming this returns a modified image
            mapsImgs.append(mapObj.title)
            selected.append(mapObj.floor)

            new_img_io = io.BytesIO()
            new_img.save(new_img_io, format='JPEG')
            new_img_content = ContentFile(new_img_io.getvalue())

            floorImg = mapImage.objects.get(title=mapObj.title + "_path")
            existing_file_path = floorImg.path.path

            if os.path.exists(existing_file_path):
                os.remove(existing_file_path)

            file_name = os.path.basename(existing_file_path)
            floorImg.path.save(file_name, new_img_content, save=True)
    for floor in selected:
        context["select"+str(floor)] = "selected"
    for title in mapsImgs:
        context[title+"_url"] = mapImage.objects.get(title=title+"_path").path.url
    buttons.update(context)
    buttons.update({'source': src})
    #new_user_input = request.POST.get('location')
    #return redirect(reverse('search_dij', kwargs={'user_input': new_user_input}))
    return render(request, "base.html", buttons)

def nearest_point(points,point):
    min_distance = math.inf
    nearest_point = None
    for pt in points:
        distance = math.sqrt((pt.pointX - point.pointX) ** 2 + (pt.pointY - point.pointY) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = pt
    return nearest_point
'''
    # Create temporary file and write image data
    with NamedTemporaryFile(suffix='.png', dir="PathIn/navApp/static/img") as temp_file:
        temp_file.write(new_img)
        temp_file.flush()  # Ensure the data is written to the file
        temp_file_url = temp_file.name
        path = default_storage.save(temp_file_url, ContentFile(new_img))
        print(temp_file_url)
        image_url = default_storage.url(path)
        temp_file.close()
# Generate temporary URL (using JsonResponse for clarity)
        # Render the template with the temporary file URL
        context = {
            'image_url': image_url
        }
        return render(request, 'base.html', context)
    '''

def serve_temp_image(request, temp_file_name):
    # Open temporary file, read content, create response
    with open(f'/navApp/Temp/{temp_file_name}', 'rb') as temp_file:
        image_data = temp_file.read()
        response = HttpResponse(content_type='image/jpeg')
        response.content(image_data)
    return response

def draw_line(image, start_point, end_point, line_color=(255, 0, 0), line_width=4):
    '''
    # Optional: Create a separate image for the shadow to allow for blurring
    shadow_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_shadow = ImageDraw.Draw(shadow_image)
    # Define the offset for the shadow
    shadow_offset = (10, 10)

    # Draw the shadow line first (slightly offset)
    start_point_shadow = (start_point[0] + shadow_offset[0], start_point[1] + shadow_offset[1])
    end_point_shadow = (end_point[0] + shadow_offset[0], end_point[1] + shadow_offset[1])
    draw_shadow.line([start_point_shadow, end_point_shadow], fill="grey", width=5)

    # Blur the shadow for a more realistic effect (optional)
    shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(5))

    # Composite the shadow onto the original image
    image.paste(shadow_image, (0, 0), shadow_image)
    '''
    draw = ImageDraw.Draw(image)
    draw.line([start_point, end_point], fill=line_color, width=line_width)

def draw_circle(image, point, circle_radius=5, circle_color=(0, 0, 255), outline_color='black', outline_width=1):
    draw = ImageDraw.Draw(image)
    # Draw the circle at the start point
    center_x, center_y = point
    # Draw the outline ellipse
    draw.ellipse([(center_x - circle_radius - outline_width, center_y - circle_radius - outline_width),
                  (center_x + circle_radius + outline_width, center_y + circle_radius + outline_width)],
                 fill=outline_color)

    # Draw the fill ellipse
    draw.ellipse([(center_x - circle_radius, center_y - circle_radius),
                  (center_x + circle_radius, center_y + circle_radius)],
                 fill=circle_color)

def creat_map(floor):
    count = Points.objects.filter(floor=floor).count()
    obj = graph.objects.get(floor=floor)
    obj.vertices = count
    obj.save()
    encoded_graph = ''
    g = Graph(count + 1)
    result = Lines.objects.filter(floor=floor).all()
    for item in result:
        x = item.startID
        if x >= 1000:
            x %= 1000
        y = item.endID
        if y >= 1000:
            y %= 1000
        length = item.Length
        g.graph[x][y] = length
        g.graph[y][x] = length
        encoded_graph = json.dumps(g.graph)
    obj.graph = encoded_graph
    obj.save()


def draw_rotated_triangle(image, center, angle):
    # Image drawing interface
    draw = ImageDraw.Draw(image)
    # Length of the triangle sides from the center
    radius = 10  # Adjust size as needed
    # Calculate triangle vertices
    # Assuming an equilateral triangle, calculate the initial unrotated vertices
    offset = math.radians(-90)  # Initial rotation to point up
    angle_rad = math.radians(angle)  # Convert angle to radians
    points = []
    center_x, center_y = center
    for i in range(3):
        # Calculate the angle for each vertex
        vertex_angle = angle_rad + i * 2 * math.pi / 3
        if (angle != 90 or angle != 270):
            vertex_angle = angle_rad + offset + i * 2 * math.pi / 3
        # Calculate the vertex coordinates
        x = center_x + radius * math.cos(vertex_angle)
        y = center_y + radius * math.sin(vertex_angle)
        points.append((x, y))
    # Draw the triangle
    draw.polygon(points, outline='red',  fill='red')
    return image
def direction_angle(point1, point2):
  """
  Calculates the direction angle between two points in a 2D Cartesian plane.

  Args:
      point1: A tuple (x1, y1) representing the first point.
      point2: A tuple (x2, y2) representing the second point.

  Returns:
      The angle in radians between the positive x-axis and the line connecting the two points.
      The angle ranges from 0 (positive x-axis) to 2*pi (counter-clockwise full circle).
  """
  x1, y1 = point1
  x2, y2 = point2
  dx = x2 - x1
  dy = y2 - y1
  return math.atan2(dy, dx)
def draw_path(org_image, path, floor):
    img = Image.open(org_image)
    img = img.resize((1680, 720))

    for p in range(len(path)):
        if p < len(path) - 1:
            obj = Points.objects.get(id=path[p]+(1000*floor))
            x = obj.pointY*10
            y = obj.pointX*10
            if x < 860:
                x -= 15
            start_point = (-1*round(x,2)+1680+15, -1*round(y,2)+720+3)
            print(start_point)
            obj = Points.objects.get(id=path[p + 1]+(1000*floor))
            x = obj.pointY * 10
            y = obj.pointX * 10
            if x < 860:
                x -= 15
            end_point = (-1*round(x,2)+1680+15, -1*round(y,2)+720+3)
            print(end_point)
            draw_line(img, start_point, end_point)
            angle = direction_angle(start_point,end_point)
            if p == 0:
                draw_circle(img,start_point, circle_color='red')
            elif p == len(path) - 2:
                draw_circle(img,end_point, circle_color='blue')
    return img
