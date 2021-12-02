from get_loader import Load_Data
from model import Face_Recognition

load_data = Load_Data('/Users/mateus/Desktop/Code/Python/pythonProject')
loaded = load_data.Load()

print(loaded[1])

test_model = Face_Recognition('/Users/mateus/Desktop/Code/Python/pythonProject/data.pt')
test_model.identify_person()