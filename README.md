# Duovigintillion translate
If you have not already looked it up, duovigintillion is the number 1 followed by 69 zeros. Kinda like google translate, but worse. Do not trust the translator under any circumstances (or google translate for that matter), this is just a learning experiment. To that end, I have not bothered keeping this codebase clean.

### Motivation
This project was mostly a playground for learning how transformer models work, so I have not put much effort into actually making this translator that good.

### Setup
Note that I developed this in WSL on a windows 11 machine with cuda support for gpu parallelisation. If you do not have the exact same machine, you will need to figure out how to make this work yourself.

1. `cd ai`
2. `python3 -m venv aienv`
3. `source aienv/bin/activate`
4. `pip install -r requirements.txt` - if this gives you errors, I pray for you and good luck.
5. `python3 app.py` - to start the backend ai web server.
6. `cd ../frontend`
7. `npm i`
8. `npm run dev`
