Flask Image Pattern Synthesizer 
is a simple web application that lets you enter a keyword, automatically
downloads the first page of Google Images for that keyword, filters the
images for relevance, analyzes them for recurring pixel arrangements in brown,
black, white and blue, and then generates and displays a transparent PNG highlighting 
the shape that appears most frequently. To set it up, clone or download the repository,
create and activate a Python 3 virtual environment, and install the dependencies with 
pip install Flask icrawler Pillow numpy tensorflow. Launch the app with flask run and 
open your browser at http://127.0.0.1:5000. Enter any keyword and submit; behind the
scenes the app uses icrawler to scrape up to 20 images, a MobileNet classifier from 
tensorflow.keras.applications to keep only images that match the keyword, Pillow and
NumPy to resize each image to 200×200 and build binary masks for our four target colors, 
and then sums those masks across all images to find pixels that occur in at least 60% 
of them. The resulting pattern is drawn as white pixels on a transparent background 
and served as pattern.png for display. The repository contains app.py with all the 
Flask routes and logic, a downloads/ folder for temporary image storage, a templates/ 
folder with the HTML form and result page, a static/ folder where the generated 
pattern.png is saved, and a requirements.txt listing the necessary Python packages. 
Simply run the application and explore
how different keywords yield unexpected abstract patterns based on color recurrence.
