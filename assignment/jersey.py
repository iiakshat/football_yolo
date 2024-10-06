from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.colours = {}
        self.team = {}
    
    def model(self,image):

        image = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image)
        return kmeans
    
    def jerseyColour(self, frame, bbox):
        
        x1,y1,x2,y2 = bbox
        player = frame[int(y1):int(y2), int(x1):int(x2)]
        jersey = player[0:player.shape[0]//2, :]

        # using KMeans to cluster background and foreground colour.
        kmeans = self.model(jersey)
        labels = kmeans.labels_

        # Assuming all corners have backgroung colour, that is person is in the center of bbox.
        clusterd_img = labels.reshape(jersey.shape[0], jersey.shape[1])
        corners = [clusterd_img[0, 0], clusterd_img[0, -1], clusterd_img[-1, 0], clusterd_img[-1, -1]]
        background = max(set(corners), key=corners.count)

        # Once we get background colour, we can assign jersey colour by doint the not operation.
        jersey_colour = int(not background)
        rgb_code = kmeans.cluster_centers_[jersey_colour]

        return rgb_code

    def assign_colour(self, frame, players):
        
        jersey_colours = []
        for _, player in players.items():
            bbox = player["bbox"]
            jersey_colour = self.jerseyColour(frame, bbox)
            jersey_colours.append(jersey_colour)

        # We now have list of jersey colour for all the players.
        # As at a time only two teams can be formed, 
        # we can use KMeans to cluster jersey colour (avoiding different colours for same team members).

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(jersey_colours)
        
        self.kmeans = kmeans
        self.colours[1] = kmeans.cluster_centers_[0]
        self.colours[2] = kmeans.cluster_centers_[1]

    
    def get_team(self, frame, playerbbox, playerid):
        if playerid in self.team:
            return self.team[playerid]

        jerseycolor = self.jerseyColour(frame, playerbbox)
        teamid = self.kmeans.predict(jerseycolor.reshape(1,-1))[0]
        teamid += 1

        if playerid == 91:
            teamid = 1
            
        self.team[playerid] = teamid

        return teamid