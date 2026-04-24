function people = getKnownPeople(knownFolder)
%get all directories in the known folder
folders = dir(knownFolder);
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name}, {'.', '..'}));

people = strings(length(folders), 1);

%loop and extract names
for i = 1:length(folders)
    people(i) = string(folders(i).name);
end
end