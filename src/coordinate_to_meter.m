function coordXY = coordinate_to_meter(coord)
%COORDINATE_TO_METER Convert geographic coordinates to approximate meters.

a = coord(1,:);
b = coord(2,:);

rangeA = max(a) - min(a);
rangeB = max(b) - min(b);

if all(abs(a) <= 90) && all(abs(b) <= 180) && ...
        rangeA < 5 && rangeB < 5

    lat = a;
    lon = b;

    latValid = lat(isfinite(lat));

    if isempty(latValid)
        lat0 = 0;
    else
        lat0 = mean(latValid);
    end

    x = lon * 111320 * cosd(lat0);
    y = lat * 111320;

    coordXY = [x; y];
else
    coordXY = coord;
end

end
