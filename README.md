# AI-generated rock songs
With thousands of distinctively different sounding rock songs produced up to now, the genre of rock music has reinvented itself over the decades. To participate in the never-ending evolution of rock music, we decided to explore the use of AI to generate uniquely new rock songs using GPT-2 for lyrics generation and TeleMelody for melody generation. 
## 1.Data Collection

## 2.Lyrics Generation
## 3.Lyric to Melody Generation
### 3.1 Preprocessing Lyrics
Preprocessing the lyrics requires for the lyrics to be in the most basic level of speech, which is a phoneme. The combination of phonemes create words.

``` 
! mkdir data/
```

``` 
From Phonemization import get_phones
get_phones('lyrics.txt')
```
### 3.2 Lyrics to Melody

- Obtain lyrics2rhythm Checkpoints from [checkpoint_best.pt](https://msramllasc.blob.core.windows.net/modelrelease/lyric2rhythm_en_best.pt)
- Obtain Template2melody Checkpoints from [checkpoint_best.pt](https://msramllasc.blob.core.windows.net/modelrelease/template2melody_best.pt)

Place each checkpoint in its respective folder, and remove placeholder checkpoint in ---->  [Input checkpoints Here](https://github.com/benfenison/Lyric-Melody-Generation/tree/main/Lyric-to-Melody/checkpoints)
