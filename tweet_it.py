from twython import Twython
from simulation import create_random_example, simulate
from animations import create_comment, single_animation


def create_content():
    while True:
        rs = create_random_example()
        try:
            r = simulate(rs, step_size=0.004)
            break
        except FloatingPointError:
            continue
    filename = single_animation(r, rs)
    comment = create_comment(rs)
    return filename, comment


def new_tweet(filename=None, status=None):
    if filename is None:
        filename, comment = create_content()
        if status is None:
            status = comment

    with open("api_key.txt") as f:
        api_data = f.readline().split(';')
    twitter = Twython(*api_data)

    video = open('./animations/{}.mp4'.format(filename), 'rb')
    response = twitter.upload_video(media=video, media_type='video/mp4')
    twitter.update_status(status=status, media_ids=[response['media_id']])


if __name__ == '__main__':
    new_tweet()
