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

            # Add the extracted words from this string to the all_words list
            all_names.extend(words)
        unique_list = set(all_names)
        all_names = unique_list
        all_names = [name for name in all_names if name != "null"]
        return JsonResponse(list(all_names), safe=False)  # Convert queryset to list for JSON response
    return render(request, "base.html")

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
    print(deGraph[227])
    queue = [(0, src)]
    print(queue)
    while queue:
        current_dist, current_node = heapq.heappop(queue)
        print("curr_dist= "+ str(current_dist))
        print("curr_node= "+ str(current_node))
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
    GroundFlr = get_object_or_404(mapImage, floor=0)
    Ground_url = GroundFlr.path.url
    firstFlr = get_object_or_404(mapImage, floor=1)
    first_url = firstFlr.path.url
    #secFlr = get_object_or_404(mapImage, floor=2)
    #sec_url = secFlr.path.url

    context = {
        'GFloor_url': Ground_url,
        '1stFloor_url': first_url,
        #'sec_url': sec_url
    }
    return render(request, "base.html", context)

def search_dij(request):
    if request.method == 'POST':
        default_imgs = mapImage.objects.filter(title__in=["1stFloor", "2ndFloor", "GFloor"])
        context = {img.title + "_url": img.path.url for img in default_imgs}
        buttons ={"select"+str(img.floor): "" for img in default_imgs}

        location = request.POST.get('location')
        destination = request.POST.get('destination')
        source = Points.objects.get(alt__icontains=location)
        goal = Points.objects.get(alt__icontains=destination)
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
                srcStair = nearest_stair(source)
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
        return render(request, "base.html", buttons)

def nearest_stair(point):
    stairs = Points.objects.filter(title="stair", floor=point.floor)
    min_distance = math.inf
    nearest_point = None
    for stair in stairs:
        distance = math.sqrt((stair.pointX - point.pointX) ** 2 + (stair.pointY - point.pointY) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = stair
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


def draw_path(org_image, path, floor):
    print(org_image)
    img = Image.open(org_image)
    img = img.resize((1680, 720))
    for p in range(len(path)):
        if p < len(path) - 1:
            obj = Points.objects.get(id=path[p]+(1000*floor))
            x = obj.pointY*10
            y = obj.pointX*10
            if x < 860:
                x -= 25
            start_point = (-1*round(x,2)+1680+15, -1*round(y,2)+720+3)
            print(start_point)
            obj = Points.objects.get(id=path[p + 1]+(1000*floor))
            x = obj.pointY * 10
            y = obj.pointX * 10
            if x < 860:
                x -= 25
            end_point = (-1*round(x,2)+1680+15, -1*round(y,2)+720+3)
            print(end_point)
            draw_line(img, start_point, end_point)
    return img
