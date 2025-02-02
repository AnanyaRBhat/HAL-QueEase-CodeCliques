from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import numpy as np

app = Flask(__name__)

# Simulating the database with a text file
data_file = 'data.txt'

if not os.path.exists(data_file):
    with open(data_file, 'w') as f:
        f.write('')

def read_data():
    with open(data_file, 'r') as f:
        data = f.readlines()
    return [line.strip().split(',') for line in data]

def write_data(data):
    with open(data_file, 'w') as f:
        for line in data:
            f.write(','.join(line) + '\n')

queue = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/customer')
def customer():
    return render_template('customer.html')

@app.route('/employee')
def employee():
    return render_template('employee.html')

@app.route('/people_count')
def people_count():
    return render_template('people_count.html')

@app.route('/api/customer/register', methods=['POST'])
def register_customer():
    name = request.json.get('name')
    phone = request.json.get('phone')
    token = len(queue) + 1
    counter = (token - 1) % 3 + 1
    status = 'Waiting'
    queue.append({'name': name, 'phone': phone, 'token': token, 'counter': counter, 'status': status})
    save_customer_data_to_file(name, phone, token, counter, status)
    estimated_time = (len(queue) - 1) * 5
    return jsonify({'token': token, 'counter': counter, 'status': status, 'estimated_time': estimated_time}), 200

def save_customer_data_to_file(name, phone, token, counter, status):
    with open(data_file, 'a') as f:
        f.write(f"{name},{phone},{token},{counter},{status}\n")

@app.route('/api/queue/status/<int:counter>', methods=['GET'])
def view_queue(counter):
    data = read_data()
    queue = [customer for customer in data if int(customer[3]) == counter]
    
    # Add estimated time to the queue based on position
    for idx, customer in enumerate(queue):
        customer.append(str(idx * 5))  # Estimated time (5 minutes per person)
    
    return jsonify(queue)

@app.route('/api/queue/update', methods=['PUT'])
def update_status():
    token = request.json.get('token')
    new_status = request.json.get('status')
    data = read_data()
    for customer in data:
        if int(customer[2]) == token:
            customer[4] = new_status
            if new_status == 'completed':
                data.remove(customer)
            elif new_status == 'pending':
                customer[4] = 'waiting'
            break
    write_data(data)
    return jsonify({'status': 'updated'}), 200

@app.route('/api/queue', methods=['GET'])
def get_all_queue():
    data = read_data()
    customers_in_queue = [customer for customer in data if customer[4] in ['waiting', 'pending']]
    return jsonify(customers_in_queue)

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("C:/Users/arb_1/Downloads/deploy.prototxt", "C:/Users/arb_1/Downloads/mobilenet_iter_73000.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    count += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text = f"People Count: {count}"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
