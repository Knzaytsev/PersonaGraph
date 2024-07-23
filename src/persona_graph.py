import json


class PersonaNode():
    def __init__(self, text: str = "", bot_id: str = "",
                 convai2_id: str = "", session_id: str = "",
                 turn: int = -1, place: int = -1,
                 split: str = "") -> None:
        self.text = text
        self.bot_id = bot_id
        self.convai2_id = convai2_id
        self.session_id = session_id
        self.turn = turn
        self.place = place
        self.split = split

    def build_id(self):
        return ':'.join([self.convai2_id, self.session_id, self.bot_id, str(self.turn), str(self.place)])

    @classmethod
    def parse_id(self, text: str = "", persona_id: str = ""):
        split, convai2_id, session_id, bot_id, turn, place = persona_id.split(':')
        return PersonaNode(text, bot_id, split + ':' + convai2_id, session_id, turn, place, split)
